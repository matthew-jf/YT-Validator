# Keep env loading first, Gunicorn/Flask import app.py as a module
from helpers import load_env, GIT_BRANCH, GIT_COMMIT
load_env(["YT_API_KEY", "OPENAI_API_KEY"])


from flask import Flask, request, jsonify, send_file
import threading
import uuid
import pandas as pd
import argparse
import os
import time
import requests 


    
app = Flask(__name__)
tasks = {}


@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'YT-Validator',
        'version': '1.0.0',
        'branch': GIT_BRANCH,
        'commit': GIT_COMMIT,
        'timestamp': time.time()
    })

@app.route('/predict', methods=['POST'])
def start_prediction():

    # TODO: Omit unnecessary humongous memory `.result` dict for every task!
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'status': 'running',
        'message': None,          # Status messages from callbacks
        'result': None,
        'error': None,
        'csv_path': None
    }

    if 'file' not in request.files:
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['error'] = 'No file part in request'
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
        # for server callback
        webhook_url=request.form.get('webhook_url'),
        pipeline_run_id=request.form.get('pipeline_run_id')  # ADD this
    )

    print(f"Received prediction request: {file.filename}", args)
    
    # Start background task
    thread = threading.Thread(target=run_prediction, args=(task_id, args))
    thread.start()
    
    return jsonify({'task_id': task_id, 'status': 'running'})

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
    
    return jsonify(task['result'].to_dict('records'))

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
        main(args, status_callback=update_status, stop_check=should_stop)

        if should_stop():
            tasks[task_id]['status'] = 'stopped'
            return
        
        # Load the CSV that was saved
        print(f"Loading results from {args.prediction_output}")
        result_df = pd.read_csv(args.prediction_output)
        
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['result'] = result_df
        tasks[task_id]['csv_path'] = args.prediction_output

        # Notify webhook of background task completion
        if hasattr(args, 'webhook_url') and args.webhook_url:
            notify_completion(args.webhook_url, task_id, args.pipeline_run_id, len(result_df))
        
    except Exception as e:
        print("Prediction error: ", e)
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['error'] = str(e)


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
