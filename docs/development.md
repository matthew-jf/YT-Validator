# Local development

## Environment

```bash
conda env create -f environment.yml
conda activate YT-Validator
```

`environment.yml` bounds pandas and numpy to AutoGluon 1.5.0's own ranges.
Do not relax those bounds: pandas 3.0 removed an argument AutoGluon still calls,
and the service crash-loops on

```
TypeError: NDFrame.fillna() got an unexpected keyword argument 'downcast'
```

AutoGluon is pinned to the version that trained the artifact.

## Secrets

```bash
cp .env.example .env
```

`YT_API_KEY` is the only required variable — a YouTube Data API key from the GCP
console. Quota is 1 unit per 50 videos, capped at 10,000 units/day, so ~500,000
availability checks a day. See [deploy.md](deploy.md#configuration) for the full
variable list.

## Get the model

The artifact is not in git.

```bash
MODEL_BUCKET=gs://jfp-yt-validator-models MODEL_VERSION=v1 ./scripts/fetch_model.sh
```

Without it, `/health` reports 503 and every prediction fails — deliberately
loudly. See [model.md](model.md).

## Run

```bash
python app.py
```

The model loads and warms *before* the port opens, so expect ~30s on a cold
start. Set `WARM_MODEL=0` to skip it when you are not scoring anything.

CLI, bypassing the service entirely:

```bash
python pipeline.py --prediction-input path/to/claims.csv
```

## Debug config

`.vscode/launch.json` — env vars optional if `.env` is present:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "YT-Validator (Conda)",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/app.py",
      "console": "integratedTerminal",
      "env": {
        "FLASK_DEBUG": "1",
        "FLASK_RUN_PORT": "3001",
        "MODEL_NAME": "ag_challenger"
      }
    }
  ]
}
```

Run `app.py` directly rather than via `flask run` — startup work (orphaned-task
reconciliation, model warm-up) happens at import, and the reloader would load
the model twice, doubling peak memory.

## Tests worth running before a deploy

```bash
python trey_pipeline/ml_pipeline/verify_deploy.py
```

Asserts a pruned artifact predicts identically to its source. Exits non-zero on
any difference, so it works as a build gate.
