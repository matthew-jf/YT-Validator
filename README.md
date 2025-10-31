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

Optional `.vscode/launch.json`:
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
        "FLASK_RUN_HOST": "0.0.0.0",
        "YT_API_KEY": "AIza...",
      },
      "args": [
        "run"
      ]
    }
  ]
}
```

2. Start the pipeline

```shell
curl -X POST http://localhost:3001/predict \
  -F "file=@$HOME/Downloads/export_unprocessed_claims_202509031526.csv" \
  -F "webhook_url=http://localhost:3000/api/ml-webhook" \
  -F "pipeline_run_id=68d88bd07c95b16053ef569a" \
  -F "skip_validation=true" 
```

Only required arg is input file.
Eg. response: note the running `task_id` returned

```json
{"status":"running","task_id":"0669d93a-22e1-4f7b-942a-89ef8ff2d836"}
```

3. Get status or results

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

## Build and deploy to Google Cloud Run

1. Build

```shell
conda env export --name YT-Validator --no-builds > environment-no-builds.yml
docker buildx build --platform linux/amd64 -t yt-validator:latest .
```

2. Env setup

```shell
export \
  PROJECT_ID=... \
  SERVICE_ACCOUNT=...

export \
  YT_API_KEY=AIza... 

# Create secret on GCP for YT API key (requires project admin)
echo -n $YT_API_KEY | gcloud secrets create youtube-api-key --data-file=-
```

3. Push to Artifact Registry

```shell
docker tag yt-validator:latest us-east1-docker.pkg.dev/$PROJECT_ID/ytdt-claims/yt-validator:latest
docker push us-east1-docker.pkg.dev/$PROJECT_ID/ytdt-claims/yt-validator:latest
  ```

4. Deploy to Cloud Run

```shell
gcloud run deploy yt-validator \
  --image us-east1-docker.pkg.dev/$PROJECT_ID/ytdt-claims/yt-validator:latest \
  --platform managed \
  --region us-east1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --service-account $SERVICE_ACCOUNT \
  --set-secrets YT_API_KEY=youtube-api-key:latest \
  --allow-unauthenticated
  ```

### Post-deployment

* Retrieve actual webservice URL

```shell
gcloud run services describe yt-validator --region us-east1 --format 'value(status.url)'
```