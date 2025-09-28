# YT-Validator


## Setup

Create conda env with expected Python / dependencies versions:

```shell
conda env create -f environment.yml
conda activate YT-Validator
```

Uncompress training data:

```shell
7z x YT.csv.7z
```

Export required envs:

```shell
YT_API_KEY=AIza...              # Grab free from GCP console - 1 unit per 50 videos capped at 10k units/day.
BASE_URL=http://localhost:3001  # Our URL, ie. YT-Validator API server
```

## CLI 

Run with inference data as only required argument:

```shell
python pipeline.py \
  --prediction-input /Users/matthew.jurewicz/Downloads/export_unprocessed_claims_202507241337.csv
  --skip-validation 
```

Fits by default `YT.csv` if found, else requires training data
eg. `--training-data /Users/matthew.jurewicz/Downloads/export_all_claims_202507241336.csv`.


## API

1. Start server:

```shell
python app.py
```

2. Start the pipeline

```shell
curl -X POST http://localhost:3001/predict \
  -F "file=@$HOME/Downloads/export_unprocessed_claims_202509031526.csv" \
  -F "webhook_url=http://localhost:3000/api/ml-webhook" \
  -F "pipeline_run_id=68d88bd07c95b16053ef569a" \
  -F "skip_validation=true" 
```

Eg. response. Note the `task_id`

```json
{"status":"started","task_id":"0669d93a-22e1-4f7b-942a-89ef8ff2d836"}
```

3. Get results using `task_id`

```shell
# Check status  
curl http://localhost:5000/status/TASK_ID

# Get JSON results
curl http://localhost:5000/results/TASK_ID

# Download CSV
curl http://localhost:5000/download/TASK_ID -o results.csv
```