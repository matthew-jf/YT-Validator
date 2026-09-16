# Keep env loading first, Gunicorn/Flask import app.py as a module
from helpers import load_env
load_env(["YT_API_KEY"])


from flask import Flask, request, jsonify, send_file
import json
from collections import Counter
from datetime import datetime, timezone
import threading
import uuid
import pandas as pd
import argparse
import os
import time
import requests 

from helpers import TaskStore, get_git_info

    
app = Flask(__name__)
tasks = TaskStore()
GIT_BRANCH, GIT_COMMIT = get_git_info()

# One scoring run saturates the available cores, so runs are serialised.
# Concurrent runs would only contend for CPU while multiplying peak memory.
INFERENCE_SLOT = threading.Semaphore(1)

# Warm the model at import (Gunicorn imports this module), so a corrupt artifact
# fails the service at startup rather than inside a background task. A missing
# artifact is a valid state -- /predict can still be called with training_data.
from pipeline import model_info, warm_model

try:
    print(f"Model warm in {warm_model():.2f}s")
except FileNotFoundError:
    print("No model artifact yet -- /predict needs training_data to build one")


@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'YT-Validator',
        'version': '1.0.0',
        'branch': GIT_BRANCH,
        'commit': GIT_COMMIT,
        'model': model_info(),
        'running_tasks': sum(1 for t in tasks.values() if t.get('status') == 'running'),
        'timestamp': time.time()
    })


@app.route('/predict', methods=['POST'])
def start_prediction():

    # Status messages from callbacks will be in `message`, and final CSV path in `csv_path`
    task_id = str(uuid.uuid4())
    tasks[task_id] = { 'status': 'running', 'message': None, 'error': None, 'csv_path': None }
    tasks.save()

    if 'file' not in request.files:
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['error'] = 'No file part in request'
        tasks.save()
        return jsonify(tasks[task_id]), 400

    # Save received file
    file = request.files['file']
    csv_path = f"data/input_{task_id}.csv"
    file.save(csv_path)

    # Extract additional args  
    args = argparse.Namespace(
        # args for pipeline.py
        prediction_input=csv_path,
        prediction_output=request.form.get('prediction_output', f"data/output_{task_id}.csv"),
        skip_validation=request.form.get('skip_validation', 'false').lower() == 'true',
        training_data=request.form.get('training_data'),  # Optional path to training data CSV
        # Monthly WESS language retrain: this run's all_claims export and the
        # reviewed verdict sheet(s), as paths on this host
        language_history=request.form.get('language_history'),
        language_eval_labels=request.form.getlist('language_eval_labels'),
        # for server callback
        webhook_url=request.form.get('webhook_url'),
        pipeline_run_id=request.form.get('pipeline_run_id')  # ADD this
    )

    print(f"Received prediction request: {file.filename}", args)
    
    # Start background task
    thread = threading.Thread(target=run_prediction, args=(task_id, args))
    thread.start()
    
    return jsonify({'task_id': task_id, 'status': 'running'})


_ARTIFACT_MEMO = {}


def _language_state():
    """Tier state from the artifact, re-read only when the file changes."""
    try:
        import predict_wess
        path = predict_wess.ARTIFACT_PATH
        if not path.exists():
            return {'tiers_live': [], 'trusted_asr_languages': [], 'artifact_trained_at': None}
        stamp = path.stat().st_mtime
        if _ARTIFACT_MEMO.get('stamp') != stamp:
            art = predict_wess.load_artifact()
            tiers = art.get('tiers', {})
            _ARTIFACT_MEMO.update(stamp=stamp, value={
                'tiers_live': [t for t in predict_wess.CASCADE if t in tiers],
                'trusted_asr_languages': sorted(tiers.get('ASR', {}).get('languages', {})),
                'artifact_trained_at': art.get('metadata', {}).get('trained_at'),
            })
        return _ARTIFACT_MEMO['value']
    except Exception as exc:
        return {'error': str(exc)}


def _count_lines(path):
    with open(path, 'rb') as handle:
        return sum(1 for _ in handle)


@app.route('/asr/status')
def asr_status():
    """Cheap, pollable view of caption collection, for the claims console.

    Reads files only: line counts, the collector's last-run summary and the
    artifact (re-parsed only when it changes). No model load, no API calls, no
    CSV parsing, so it stays cheap as the cache grows. `remaining` is as of the
    last collector run, not recomputed per request.
    """
    import predict_wess
    state = {'queue': {}, 'cache': {}, 'collector': {},
             'languages': _language_state(),
             'version': {'branch': GIT_BRANCH, 'commit': GIT_COMMIT}}

    queue_path = ASR_QUEUE_PATH
    if os.path.exists(queue_path):
        with open(queue_path, encoding='utf-8', errors='replace') as handle:
            header = [c.strip() for c in (handle.readline() or '').split(',')]
        state['queue'] = {
            'rows': max(0, _count_lines(queue_path) - 1),
            'written_at': datetime.fromtimestamp(os.path.getmtime(queue_path), tz=timezone.utc).isoformat(timespec='seconds'),
            'has_licensed': 'licensed' in header,
            'has_triage': 'triage' in header,
        }

    cache_path = predict_wess.ASR_CACHE_PATH
    if os.path.exists(cache_path):
        cache = predict_wess.load_asr_cache(cache_path)
        with_track = sum(1 for code in cache.values() if code)
        # Per-language video counts, by base code (es-419 counts as es) - the same
        # grouping certification uses. "" is left out: it is already no_track.
        languages = Counter(predict_wess.asr_base(code) for code in cache.values() if code)
        state['cache'] = {'videos': len(cache), 'with_track': with_track,
                          'no_track': len(cache) - with_track,
                          'languages': dict(languages.most_common())}

    status_path = predict_wess.ASR_STATUS_PATH
    if os.path.exists(status_path):
        try:
            state['collector'] = json.loads(open(status_path, encoding='utf-8').read())
        except ValueError as exc:
            state['collector'] = {'error': f'unreadable status file: {exc}'}
    return jsonify(state)


@app.route('/asr/queue', methods=['POST'])
def refresh_asr_queue():
    """Replace the daily ASR collector's queue with the latest unprocessed claims.

    Called by ytdt-claims-pipeline after each claims ingestion. Runs no model and
    makes no API calls. Rows keep the triage the last /predict gave them where
    video ids match; newer rows have none, so the collector orders them by views.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    try:
        rows = merge_asr_queue(pd.read_csv(request.files['file'], low_memory=False))
    except ValueError as exc:   # includes pandas parse errors and an empty file
        return jsonify({'error': str(exc)}), 400
    return jsonify({'status': 'ok', 'rows': rows})


@app.route('/status/<task_id>')
def get_status(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify({
        'status': tasks[task_id]['status'],
        'error': tasks[task_id]['error']
    })


@app.route('/results/<task_id>')
def get_results(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = tasks[task_id]
    if task['status'] != 'completed':
        return jsonify({'error': 'Task not completed'}), 400
    
    if not task.get('csv_path'):
        return jsonify({'error': 'No results available'}), 400
    
    return jsonify(pd.read_csv(task['csv_path']).to_dict('records'))


@app.route('/download/<task_id>')
def download_csv(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = tasks[task_id]
    if task['status'] != 'completed':
        return jsonify({'error': 'CSV not available'}), 400
    
    return send_file(task['csv_path'], as_attachment=True)


@app.route('/stop/<task_id>', methods=['POST'])
def stop_task(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = tasks[task_id]
    if task['status'] not in ['running']:
        return jsonify({'error': f'Task already {task["status"]}'}), 400
    
    tasks[task_id]['stopped'] = True
    tasks.save()
    return jsonify({'status': 'stopping', 'task_id': task_id})



def run_prediction(task_id, args):

    start_time = time.time()

    def update_status(message):              
        elapsed = int(time.time() - start_time)
        current_time = time.strftime("%H:%M:%S")
        print(f"[{current_time}] {message} (elapsed: {elapsed}s)")
        tasks[task_id]['message'] = message

    def should_stop():
        return tasks[task_id].get('stopped', False)

    try:
        
        from pipeline import main
        with INFERENCE_SLOT:                      # one scoring run at a time
            main(args, status_callback=update_status, stop_check=should_stop)
            if not should_stop():
                add_language_predictions(args.prediction_output, update_status,
                                         history=args.language_history,
                                         eval_labels=args.language_eval_labels)

        if should_stop():
            tasks[task_id]['status'] = 'stopped'
            tasks.save()
            return
        
        # Task completed. Load the CSV that was saved
        result_df = pd.read_csv(args.prediction_output)
        update_status(f"Completed ({len(result_df)} rows) → {args.prediction_output}")

        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['csv_path'] = args.prediction_output
        tasks.save()

        # Notify webhook of background task completion
        if hasattr(args, 'webhook_url') and args.webhook_url:
            notify_completion(args.webhook_url, task_id, args.pipeline_run_id, len(result_df))
        
    except Exception as e:
        print("Prediction error: ", e)
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['error'] = str(e)
        tasks.save()
        

# The scored batch the daily ASR collector works through (predict_wess.py --collect-asr)
ASR_QUEUE_PATH = 'data/asr_queue.csv'


def write_asr_queue(df, queue_path=ASR_QUEUE_PATH):
    """Replace the collector's queue atomically, so it never reads a half-written file."""
    tmp_path = f'{queue_path}.tmp'
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, queue_path)


def merge_asr_queue(incoming, queue_path=ASR_QUEUE_PATH):
    """Make `incoming` the collector's queue, carrying over triage it already knows."""
    if 'video_id' not in incoming.columns:
        raise ValueError('queue file has no video_id column')
    if 'triage' not in incoming.columns and os.path.exists(queue_path):
        previous = pd.read_csv(queue_path, low_memory=False,
                               usecols=lambda column: column in ('video_id', 'triage'))
        if 'triage' in previous.columns:
            known = (previous.dropna(subset=['triage'])
                     .drop_duplicates('video_id', keep='last')
                     .set_index('video_id')['triage'])
            incoming = incoming.assign(triage=incoming['video_id'].map(known))
    write_asr_queue(incoming, queue_path)
    return len(incoming)


def retrain_languages(predict_wess, history, eval_labels, status):
    """Monthly language retrain. On any problem, keep the previous artifact and say so.

    Either verdict sheet may be absent, and a listed path that doesn't exist is
    skipped: retraining needs the all_claims export and at least one sheet.
    """
    if not os.path.exists(history):
        status(f'Language retrain skipped: language_history not found ({history})')
        return
    paths = predict_wess.label_paths(eval_labels)
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        status(f'Language retrain: verdict sheet(s) not found, ignored: {missing}')
    present = [path for path in paths if os.path.exists(path)]
    if not present:
        status('Language retrain skipped: no readable language_eval_labels')
        return
    try:
        predict_wess.train_from_exports(history, present, status)
    except Exception as exc:
        status(f'Language retrain failed, keeping the previous artifact ({exc})')

def add_language_predictions(output_path, status, history=None, eval_labels=None):
    """Add WESS language columns to a scored claims file, in place.

    Best effort by design: the verdict output is the deliverable, so a missing
    language artifact or a language-model error is logged and the file is left
    as the verdict model wrote it. Makes no YouTube API calls - ASR answers come
    only from the collector's cache. When `history` is given (monthly), the
    language model is retrained from the pipeline's exports first.
    """
    try:
        scored = pd.read_csv(output_path, low_memory=False)
        # Hand the batch to the collector first, so a failure below cannot stop
        # the caption lookups that next month's certification depends on.
        write_asr_queue(scored)

        import predict_wess
        if history:
            retrain_languages(predict_wess, history, eval_labels, status)
        elif eval_labels:
            status('Language retrain skipped: language_eval_labels given without language_history')
        if not predict_wess.ARTIFACT_PATH.exists():
            status('Language prediction skipped: no wess_artifact.json')
            return
        out = predict_wess.predict(scored, predict_wess.load_artifact(), status)
        out.to_csv(output_path, index=False)
    except Exception as exc:
        status(f'Language prediction skipped ({exc})')

def notify_completion(webhook_url, task_id, pipeline_run_id, num_results):
    print(f"Notifying the Webhook at: {webhook_url}" )

    if webhook_url and task_id in tasks:
        task = tasks[task_id]
        if task['status'] == 'completed':

             # csv_path hits CSV download route above
            payload = {
                'task_id': task_id,
                'status': task['status'],
                'error': task['error'],
                'csv_path': f"/download/{task_id}", 
                'num_results': num_results,
                'pipeline_run_id': pipeline_run_id
            }

            # Call webhook. Raise iff 4xx/5xx: webhook failure shouldn't break ML task
            try:
                response = requests.post(webhook_url, json=payload, timeout=10)
                response.raise_for_status() 
                print(f"Webhook notification successful: {response.status_code} - {response.json()}")
            except requests.exceptions.RequestException as e:
                print(f"Webhook notification failed: {e}")


if __name__ == '__main__':
    
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("FLASK_RUN_PORT", "3001")))  # So PORT env also works
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)
