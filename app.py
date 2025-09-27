from flask import Flask, request, jsonify, send_file
import threading
import uuid
import pandas as pd
import argparse
import os
import time

app = Flask(__name__)

tasks = {}


@app.route('/predict', methods=['POST'])
def start_prediction():

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'status': 'running',
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
        prediction_input=csv_path,
        prediction_output=request.form.get('prediction_output', f"data/output_{task_id}.csv"),
        skip_validation=request.form.get('skip_validation', 'false').lower() == 'true'
    )

    print(f"Received prediction request: {file.filename}", args)
    
    # Start background task
    thread = threading.Thread(target=run_prediction, args=(task_id, args))
    thread.start()
    
    return jsonify({'task_id': task_id, 'status': 'started'})

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


def run_prediction(task_id, args):

    start_time = time.time()

    def update_status(status):              
        elapsed = int(time.time() - start_time)
        current_time = time.strftime("%H:%M:%S")
        print(f"[{current_time}] {status} (elapsed: {elapsed}s)")
        tasks[task_id]['status'] = status

    try:
        
        from pipeline import main
        main(args, status_callback=update_status)
        
        # Load the CSV that was saved
        print(f"Loading results from {args.prediction_output}")
        result_df = pd.read_csv(args.prediction_output)
        
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['result'] = result_df
        tasks[task_id]['csv_path'] = args.prediction_output
        
    except Exception as e:
        print("Prediction error: ", e)
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['error'] = str(e)


if __name__ == '__main__':
    
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", 3001))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host=host, port=port, debug=debug)