# YT-Validator


## Setup

Create conda env with expected Python / dependencies versions:

```shell
conda env create -f environment.yml
conda activate YT-Validator
```

On servers where systemd runs the service as `root`, update the env as root so changes land in the right place:

```shell
sudo /opt/miniconda3/bin/conda env update -n YT-Validator -f /opt/yt-validator/environment.yml --prune
sudo systemctl restart yt-validator
```

Configure secrets via `.env` `cp .env.example .env` and edit .env with your keys
Or export required envs:

```shell
YT_API_KEY=AIza...                # Grab free from GCP console - 1 unit per 50 videos capped at 10k units/day.
```

## Model

Predictions come from the pretrained AutoGluon challenger stack in
`trey_pipeline/models/ag_challenger` (with its calibrated Human-Review
threshold in `trey_pipeline/models/ag_threshold.json`). No training happens at
runtime; to retrain, see `trey_pipeline/Machine_Learning_Pipeline.md`.

## CLI 

Run with inference data as only required argument:

```shell
python pipeline.py \
  --prediction-input /Users/matthew.jurewicz/Downloads/export_unprocessed_claims_202507241337.csv
```


## API

1. Start server:

```shell
python app.py
```

2. Verify it's up:

```shell
curl http://localhost:3001/health
```

Response includes git branch/commit for deploy verification:

```json
{"branch":"chore/cli-api-wrapper","commit":"be0faa3","service":"YT-Validator","status":"healthy","timestamp":1778717209.804979,"version":"1.0.0"}
```

3. Start the pipeline

```shell
curl -X POST http://localhost:3001/predict \
  -F "file=@$HOME/Downloads/export_unprocessed_claims_202509031526.csv" \
  -F "webhook_url=http://localhost:3000/api/ml-webhook" \
  -F "pipeline_run_id=68d88bd07c95b16053ef569a"
```

Only required arg is input file.
Eg. response: note the running `task_id` returned

```json
{"status":"running","task_id":"0669d93a-22e1-4f7b-942a-89ef8ff2d836"}
```

4. Get status or results

Follow up with `task_id` from previous step:

```shell
# Check status  
curl http://localhost:3001/status/TASK_ID

# Get JSON results
curl http://localhost:3001/results/TASK_ID

# Download CSV
curl http://localhost:3001/download/TASK_ID -o results.csv

# Stop task
curl -X POST http://localhost:3001/stop/TASK_ID

```

## Debug config (optional)

`.vscode/launch.json` — envs optional if using `.env`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Flask (Conda Debug)",
      "type": "python",
      "request": "launch",
      "module": "flask",
      "env": {
        "FLASK_APP": "app.py",
        "FLASK_ENV": "development",
        "FLASK_DEBUG":  "1",
        "FLASK_RUN_PORT": "3001",
        "FLASK_RUN_HOST": "0.0.0.0"
      },
      "args": ["run"]
    }
  ]
}
```

## Deploy (systemd)

On servers where systemd runs the service as `root`, update the env as root so changes land in the right place:

```shell
sudo /opt/miniconda3/bin/conda env update -n YT-Validator -f /opt/yt-validator/environment.yml --prune
sudo systemctl restart yt-validator
```

Tail logs:

```shell
sudo journalctl -u yt-validator -f
```

Verify deployment via health check — response includes git branch/commit:

```shell
curl http://localhost:3001/health
```