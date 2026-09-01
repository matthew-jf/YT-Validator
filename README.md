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
- `triage` — `AUTO_N_LICENSED` / `AUTO_N_UNAVAILABLE` / `AUTO_N` / `AUTO_Y` (auto-decidable at >= 95% holdout accuracy) or `REVIEW` (route to manual review)


## WESS language prediction

`predict_wess.py` predicts the WESS `language_id` (the number entered in the
monthly sheet, `WESS_LAN_num` in `sheets_language_families.csv`) for
unprocessed claims. Precision-first cascade — a tier only fires when its
cutoff, calibrated to >= 95% precision, is met; everything else is `REVIEW`:

1. `CHANNEL` — the channel's labeled history is unanimous (min claim count is tuned)
2. `TITLE` — the title contains a validated language-name rule (Anglicized
   names from the sheets mapping plus native-name aliases mined from history,
   e.g. "bahasa melayu jambi")
3. `FASTTEXT` — supervised fastText classifier over channel-prior tokens
   (channel's top historical languages, leave-one-out at fit time) + title text
4. `LID` — pretrained lid.176 language ID on the title, ISO -> WESS via the
   sheets mapping, ambiguous codes resolved by history frequency

Train (builds `wess_artifact.json` + `wess_fasttext.ftz`; delete both to
retrain) and evaluate against a completed monthly sheet:

```shell
python predict_wess.py \
  --prediction-input unprocessed_claims.csv \
  --history all_claims.csv \
  --eval-labels "Unprocessed_ClaimsMCN_Matt_JULY(1).csv"
```

When `--eval-labels` is present at training time, per-tier cutoffs are
calibrated on a stable half of that reviewed batch (rows whose video_id also
appears in history are excluded from training) and scored on the other half —
same recalibration philosophy as the verdict model's triage cutoffs. July 2026
batch, held-out half: `CHANNEL` 24/24 = 100%, `FASTTEXT` 124/128 = 96.9%,
total 97.4% accuracy at 28.3% coverage of language-labeled claims. `TITLE` and
`LID` did not meet the precision bar on that batch and disabled themselves.

Predict-only runs reuse the cached artifacts (~1s for a monthly batch, no
`--history` needed):

```shell
python predict_wess.py --prediction-input unprocessed_claims.csv
```

Output CSV = input columns plus:

- `predicted_language_id` — WESS number, empty when routed to review
- `predicted_language_name` — Anglicized name from the sheets mapping
- `language_source` — `CHANNEL` / `TITLE` / `FASTTEXT` / `LID` / `REVIEW`
- `language_confidence` — 1.0 for exact-rule tiers, model probability otherwise

`lid.176.ftz` is downloaded automatically on first use. fastText comes from
pip (see `environment.yml`); on macOS, if the source build fails or predict
raises a numpy-2 copy error, `pip install fasttext-wheel "numpy<2"`.


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