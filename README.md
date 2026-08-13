# YT-Validator

Scores YouTube copyright claims with a pretrained AutoGluon model and assigns
each one of three actions — **Auto Yes**, **Auto No**, or **Human Review** —
so only genuine edge cases reach a person.

Runs as an HTTP service on port `3001`, called by
[ytdt-claims-pipeline](https://github.com/JesusFilm/ytdt-claims-pipeline) during
its ML enrichment step. Inference only; no training happens at runtime.

```
claims CSV ──▶ POST /predict ──▶ task_id ──▶ webhook ──▶ enriched CSV
                    │                                    (rating + action)
                    └── feature engineering ──▶ AutoGluon ──▶ decision rules
```

## Documentation

| | |
|---|---|
| [docs/development.md](docs/development.md) | local setup, running, debugging |
| [docs/api.md](docs/api.md) | endpoints, webhook contract, telemetry |
| [docs/model.md](docs/model.md) | decision rules, artifact, benchmarks, caveats |
| [docs/deploy.md](docs/deploy.md) | **deploy runbook** — publish, update, roll back, troubleshoot |
| [trey_pipeline/](trey_pipeline/Machine_Learning_Pipeline.md) | training pipeline |

## Quick start

```bash
conda env create -f environment.yml && conda activate YT-Validator
cp .env.example .env                       # set YT_API_KEY
MODEL_BUCKET=gs://jfp-yt-validator-models MODEL_VERSION=v1 ./scripts/fetch_model.sh
python app.py
```

The model artifact (~1.4 GB) is **not in git** — it is versioned in GCS and
fetched by `scripts/fetch_model.sh`. See [docs/model.md](docs/model.md#why-the-artifact-is-not-in-git).

Verify:

```bash
curl -s localhost:3001/health
```

Returns 200 only once the model is loaded and warm, 503 otherwise. Cold start is
~30s, since the model loads before the port opens.

Score a file:

```bash
curl -X POST http://localhost:3001/predict -F "file=@claims.csv"
```

## Layout

```
app.py                        HTTP service, task lifecycle, webhooks
pipeline.py                   inference: features -> model -> decision rules
scripts/fetch_model.sh        pull the artifact from GCS
scripts/upload_model.sh       publish a new model version
trey_pipeline/ml_pipeline/    feature_utils, train, export_deploy, verify_deploy
trey_pipeline/models/         artifacts (ag_challenger/ is gitignored)
```

`feature_utils.py` is shared by training and inference so the two cannot drift.
