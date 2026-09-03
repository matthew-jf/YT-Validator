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

## Model

The pipeline caches the fitted model (plus tuned decision threshold and triage
cutoffs) to `model.joblib`. If the artifact exists, training is skipped. Delete
it to retrain — training then requires `--training-data` (a claims export with
`verdict`, and optionally `no_code`).

Training details: claims with rule-driven verdicts (`no_code` L/V/N/X) are
excluded, claims from 2022+ are used, and the most recent ~5 months are held
out to tune the Y/N decision threshold (max balanced accuracy) and the triage
cutoffs (auto buckets held to >= 95% accuracy on the holdout).

Note on triage cutoffs: holdout-calibrated cutoffs proved optimistic on a real
unprocessed batch (holdout claims are easier than monthly leftovers). The
shipped `model.joblib` has cutoffs recalibrated on the July 2026 reviewed batch
(AUTO_N/AUTO_Y both ~97% accurate there). After retraining, prefer
recalibrating cutoffs against the most recent reviewed batch by comparing a
prediction run's raw probabilities to the human verdicts.

## Channel verdict history (AUTO_N_CHANNEL)

`channel_verdicts.csv` (bundled, like `Licensed.csv`) holds each channel whose
labeled Y/N verdicts are unanimous, with the claim count. It is derived from a
labeled claims export via `build_channel_verdicts()` and regenerated
automatically whenever the model is retrained; the raw export itself is not
committed. Rule-driven verdicts (`no_code` L/V/N/X) are excluded when the
export carries `no_code`.

At predict time every row gets `channel_history_verdict` /
`channel_history_claims` for reviewers, and a REVIEW row is upgraded to
`AUTO_N_CHANNEL` only when all of:

- the channel's history is unanimously N with >= `MIN_CHANNEL_CLAIMS` (3) claims
- the model also strongly leans N (`rating` <= `CHANNEL_RATING_CAP`, 0.025)

Both constants were calibrated on the July 2026 reviewed batch, where the
bucket is 95.4% accurate (65 claims, ~2.5% of the REVIEW queue). The gate is
deliberately strict: unanimous channel history alone was only ~83% accurate
there, and unanimous-Y history was <70%, so Y never auto-decides. Existing
buckets are untouched (only REVIEW rows are upgraded). Like the triage
cutoffs, re-check `MIN_CHANNEL_CLAIMS` / `CHANNEL_RATING_CAP` against the next
reviewed batch — the operating point sits close to the 95% target.

Configure secrets via `.env` `cp .env.example .env` and edit .env with your keys
Or export required envs:

```shell
YT_API_KEY=AIza...                # Grab free from GCP console - 1 unit per 50 videos capped at 10k units/day.
```

## CLI 

Run with inference data as only required argument:

```shell
python pipeline.py \
  --prediction-input /Users/matthew.jurewicz/Downloads/export_unprocessed_claims_202507241337.csv
  --skip-validation 
```

Uses cached `model.joblib` if found, else requires training data
eg. `--training-data /Users/matthew.jurewicz/Downloads/export_all_claims_202507241336.csv`.

If the input CSV already has a `video_available` column it is reused and the
YouTube API is not called (no `YT_API_KEY` needed for such offline runs).

Output CSV = input columns plus:

- `licensed`, `media_component_id`, `video_available` — enrichment (as before)
- `rating` — model probability of verdict Y, forced to 0 for licensed assets and unavailable videos (as before)
- `predicted_verdict` — Y/N at the tuned threshold, after the licensed/unavailable rules
- `confidence` — max(p, 1-p) of the raw model probability
- `channel_history_verdict`, `channel_history_claims` — the channel's unanimous historical verdict and labeled claim count (empty/0 if unseen or mixed)
- `triage` — `AUTO_N_LICENSED` / `AUTO_N_UNAVAILABLE` / `AUTO_N` / `AUTO_Y` / `AUTO_N_CHANNEL` (auto-decidable at >= 95% reviewed-batch accuracy) or `REVIEW` (route to manual review)


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
  -F "pipeline_run_id=68d88bd07c95b16053ef569a" \
  -F "skip_validation=true" 
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