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

## CLI 

Run with inference data as only required argument:

```shell
python pipeline.py --prediction-input /Users/matthew.jurewicz/Downloads/export_unprocessed_claims_202507241337.csv
```

Fits by default `YT.csv` if found, else requires training data
eg. `--training-data /Users/matthew.jurewicz/Downloads/export_all_claims_202507241336.csv`.


## API

1. Start server

```shell
python app.py
```

2. Start the pipeline

```shell
curl -X POST http://localhost:3001/predict \
  -H "Content-Type: application/json" \
  -d '{"prediction_input": "~/Downloads/export_unprocessed_claims_202509031526.csv"}'
```

3. Get results  

```shell
# Check status  
curl http://localhost:5000/status/TASK_ID

# Get JSON results
curl http://localhost:5000/results/TASK_ID

# Download CSV
curl http://localhost:5000/download/TASK_ID -o results.csv
```