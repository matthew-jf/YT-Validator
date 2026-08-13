# Training pipeline

Trains the model that `pipeline.py` serves. Nothing here runs in production —
inference loads a pretrained artifact. For what gets served and how, see
[docs/model.md](../docs/model.md).

## Components

| file | |
|---|---|
| `ml_pipeline/feature_utils.py` | duration maths and fuzzy title matching. **Shared with inference** so training and serving transforms cannot drift. |
| `ml_pipeline/train.py` | trains the XGBoost baseline and AutoGluon ensemble, calibrates the review threshold, writes to `models/` |
| `ml_pipeline/export_deploy.py` | prunes a trained artifact to an inference-only clone (~3.1 GB → ~1.4 GB) |
| `ml_pipeline/verify_deploy.py` | asserts a pruned artifact predicts identically to its source |
| `ml_pipeline/grade_against_verdicts.py` | grades a claims export against human verdicts — the production-format counterpart to `visualize.py` |
| `ml_pipeline/visualize.py` | scorecard against human-adjudicated ground truth |
| `pipeline.sh` | orchestrator for the train → infer → visualize loop, via `uv` |

`pipeline.sh` and the PEP 723 inline dependencies exist for local
experimentation. **Production does not use them** — it runs the conda env from
`environment.yml`, which pins AutoGluon to the version that trained the
artifact. Keep the two in step when changing dependencies.

## Data

| file | |
|---|---|
| `data/all_claims.csv` | ~528K adjudicated claims (`verdict` `Y`/`N`), 225 MB, Git LFS |
| `data/unprocessed_claims.csv` | claims awaiting a decision (`verdict = 'U'`) |
| `data/unprocessed_claims_Matt_MCN_MARCH1Rev.csv` | human verdicts for those — the evaluation ground truth |

Both claim files come from MySQL views built outside this repo, partitioned by
verdict, so training data and production input are disjoint by construction:

```sql
-- export_all_claims
FROM youtube_mcn_claims WHERE asset_labels = "Jesus Film Media";

-- export_unprocessed_claims
FROM youtube_mcn_claims WHERE verdict = 'U' AND asset_labels = "Jesus Film Media";
```

`train.py` reads `data/all_claims.csv` by default; pass `data_path` to
`run_training()` to use another file.

## Train

```bash
cd trey_pipeline
./pipeline.sh
```

Each step is skipped when its output already exists — including inference. To
re-run against new data, remove or rename the previous `data/output_claims.csv`
first, or the pipeline will silently skip scoring.

To publish the result, see
[docs/deploy.md](../docs/deploy.md#publish-a-model-version).

## Two models

**XGBoost** (`models/xgb_baseline.json`) — a lightweight baseline on the numeric
features, weighted 3× against false negatives. Used to derive a target
false-negative count for calibration.

**AutoGluon** (`models/ag_challenger/`) — `presets='high_quality'` over
LightGBM, CatBoost and RandomForest variants, ensembled and refit on full data.
This is what production serves.

The preset decides which model serves: it enables refit-on-full-data and points
best-model at the refit variant, frozen into the artifact as `model_best`.
Nothing in `pipeline.py` names a model.

## Evaluation

Use `grade_against_verdicts.py`, which takes any claims export plus its
reviewed verdicts file, scores it, applies the production decision rules and
reports the comparison:

```bash
python ml_pipeline/grade_against_verdicts.py <claims.csv> <verdicts.csv>
```

It reuses the `licensed` / `video_available` already in the export instead of
re-checking YouTube, so an old snapshot is graded as it stood at the time
rather than against today's availability.

Report **balanced accuracy**, not accuracy — positives are a minority, so
accuracy is dominated by the majority class.

`visualize.py` does something similar and adds charts, but reads `AG_Action`,
which only `inference.py` produces. It cannot grade production output, where
the columns are `rating` / `action`.

> **The March benchmark flattered the model.** Graded against July's 4,470
> human verdicts — a population with no overlap at all with training — it
> auto-rejects **36% of valid claims** (403 of 1,108), against 6.6% on
> MARCH1Rev. AUC falls from 0.91 to 0.80, and the median score for a valid
> claim drops from 0.93 to 0.45. Only 39 of those misses come from the
> licensed/unavailable rule; the rest are the model. Treat July as the
> realistic number.

> **`train.py`'s own train/test split is not a clean holdout.** 26.3% of its test
> rows share an exact feature-vector with a training row — nine coarse features
> over ~1,085 assets make distinct claims indistinguishable to the model. Those
> rows score 0.009 log loss against 0.113 for the rest, so any metric from that
> split is optimistic. There is no dedup and no grouped split. Production input
> does not have this problem (0 of 7,640 incoming claims collide), which is why
> the MARCH1Rev benchmark is the one to trust.

> **The calibrated threshold is degenerate.** The search minimises false
> negatives with no cost on review volume, and false negatives fall
> monotonically as the threshold drops — so it always returns the floor of its
> range (`0.1`). A two-sided objective is needed for the number to mean
> anything.
