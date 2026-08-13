# HTTP API

Flask service on port `3001`. Scoring is asynchronous: `/predict` returns
immediately with a `task_id` and the work continues on a background thread.
Callers either poll `/status` or supply a `webhook_url`.

Runs are serialised — one scoring job saturates the available cores, so a second
request waits for the first rather than contending with it.

## `GET /health`

Readiness, not liveness. Returns **200 only when the model is loaded and warm**,
**503** otherwise, so a broken artifact cannot masquerade as a working service.

```bash
curl -s localhost:3001/health
```

```json
{
  "status": "healthy", "branch": "ML-Pipeline", "commit": "0f5ee69",
  "service": "YT-Validator", "version": "1.0.0", "running_tasks": 0,
  "model": {
    "model_best": "WeightedEnsemble_L2_FULL", "loaded": true, "warm": true,
    "load_seconds": 0.26, "warm_seconds": 30.21, "peak_rss_mb": 1632.1,
    "auto_yes_threshold": 0.97, "review_threshold": 0.1,
    "model_dir": "/opt/yt-validator/trey_pipeline/models/ag_challenger"
  }
}
```

`branch` and `commit` identify the deployed code; `model_best` identifies the
serving model.

## `POST /predict`

| field | required | |
|---|---|---|
| `file` | yes | CSV of claims to score |
| `webhook_url` | no | POSTed on completion |
| `pipeline_run_id` | no | echoed back in the webhook, for correlation |
| `prediction_output` | no | output path (default `data/output_<task_id>.csv`) |

```bash
curl -X POST http://localhost:3001/predict \
  -F "file=@unprocessed_claims.csv" \
  -F "webhook_url=http://localhost:3000/api/ml-webhook" \
  -F "pipeline_run_id=68d88bd07c95b16053ef569a"
```

```json
{"status": "running", "task_id": "0669d93a-22e1-4f7b-942a-89ef8ff2d836"}
```

The input CSV must contain `asset_id`, `video_title`, `asset_title`,
`video_duration_sec`, `duration_seconds`, `matching_duration`, `longest_match`.
Missing columns fail the task immediately. A `video_id` column additionally
triggers a YouTube availability check (1 quota unit per 50 rows).

Output is the input plus `rating`, `action`, `licensed`, `media_component_id`
and `video_available`. See [model.md](model.md) for how `action` is derived.

## `GET /status/<task_id>`

```json
{"status": "running", "error": null}
```

`running` → `completed` | `failed` | `stopped`. A task interrupted by a restart
is failed at startup rather than left `running` forever.

## `GET /results/<task_id>`

Predictions as JSON. 400 unless the task completed.

## `GET /download/<task_id>`

The output CSV. This is the path the webhook advertises.

## `POST /stop/<task_id>`

Cooperative cancellation — the run checks between stages and stops at the next
one, so it is not instant.

## Webhook

On completion the service POSTs to `webhook_url`:

```json
{
  "task_id": "0669d93a...", "status": "completed", "error": null,
  "csv_path": "/download/0669d93a...", "num_results": 7640,
  "pipeline_run_id": "68d88bd07c95b16053ef569a"
}
```

Fetch the results from `csv_path` on this service. A webhook failure is logged
but never fails the task — the results remain available via the endpoints above.

## Telemetry

Completed tasks carry a `telemetry` block in `data/tasks.json`, also logged as
`[telemetry] <task_id>: {...}`:

```json
{
  "rows": 7640, "model_load_seconds": 0.28, "predict_seconds": 2.47,
  "total_seconds": 2.87, "peak_rss_mb": 1343.4,
  "actions": {"Auto No": 4465, "Human Review": 2852, "Auto Yes": 323}
}
```
