# The model

Predictions come from a pretrained AutoGluon ensemble. **No training happens at
runtime** — `pipeline.py` only loads and scores. Training lives in
[`trey_pipeline/`](../trey_pipeline/Machine_Learning_Pipeline.md).

## Decision rules

`pipeline.py` turns one probability (`rating`) into one of three actions,
applying rules top to bottom — later rules override earlier ones.

| # | Rule | Condition | Result |
|---|---|---|---|
| 1 | High confidence | `rating >= 0.97` | Auto Yes |
| 2 | Review band | `0.1 <= rating < 0.97` | Human Review |
| 3 | Low confidence | `rating < 0.1` | Auto No |
| 4 | Exact-duration rescue | Auto No **and** `duration_diff_sec == 0` | → Human Review |
| 5 | Hard reject | `licensed` **or** not `video_available` | → Auto No, `rating = 0.0` |

Rule 5 overrides everything, including Auto Yes, and involves no model — it is
driven by `Licensed.csv` and a live YouTube availability check.

Only **Auto Yes** and **Auto No** can be silently wrong; Human Review is seen by
a person. A false negative — a claim whose true verdict is `Y` that was sent to
Auto No — is the error the design most tries to avoid, and `train.py` weights it
3× accordingly.

The `0.97` cutoff is hardcoded in `pipeline.py`; the review threshold is read
from `trey_pipeline/models/ag_threshold.json`.

> **Known issue.** The stored threshold (`0.1`) is not meaningfully calibrated.
> `train.py`'s search minimises false negatives with no cost on review volume,
> and false negatives fall monotonically as the threshold drops — so it always
> returns the floor of its search range. Fixing it needs a two-sided objective.

## Features

`feature_utils.py` is shared by training and inference so transforms cannot
drift. Nine inputs: duration difference and ratio, three fuzzy title-match
scores (`thefuzz`: ratio, token-sort, token-set), `matching_duration`,
`longest_match`, and the two raw titles.

## Artifact layout

```
trey_pipeline/models/
  ag_challenger/          # served artifact — NOT in git, fetched from GCS
  ag_challenger_deploy/   # local export output — gitignored
  ag_threshold.json       # review threshold
  xgb_baseline.json       # baseline model, unused at inference
```

The training output is ~3.1 GB. Most of it is never read at inference: unused
candidate models, cached training data (`utils/data/X.pkl`, ~279 MB) and
out-of-fold predictions. `export_deploy.py` strips those:

```bash
python trey_pipeline/ml_pipeline/export_deploy.py
python trey_pipeline/ml_pipeline/verify_deploy.py   # must print PASS
```

**3,082.9 MB → 1,382.4 MB (55%)**, predictions bit-identical — `verify_deploy.py`
scores the same rows through both artifacts and compares raw probabilities and
final actions, exiting non-zero on any difference.

What survives is the serving path for `WeightedEnsemble_L2_FULL`:

| | |
|---|---|
| RandomForestEntr_BAG_L1_FULL | 687 MB |
| RandomForestGini_BAG_L1_FULL | 687 MB |
| LightGBM_BAG_L1_FULL | 7.7 MB |
| WeightedEnsemble_L2_FULL + scaffolding | ~2 MB |

The two RandomForests are 99% of it — unbounded-depth trees over nine features.

## Why the artifact is not in git

GitHub caps files at 100 MB and the RandomForest files are ~687 MB each. It was
tracked in Git LFS, which failed badly in practice: a clone on a host without
`git-lfs` yields ~130-byte pointer stubs, the service starts, reports healthy,
and every prediction fails.

It now lives in GCS, versioned by prefix, fetched by `scripts/fetch_model.sh` —
which rejects pointer stubs rather than trusting whatever is on disk. See
[deploy.md](deploy.md#publish-a-model-version).

## Which model serves

Chosen at *train* time, not in code. `presets='high_quality'` in `train.py`
enables refit-on-full-data and points best-model at the refit variant; that
choice is frozen into the artifact as `model_best`. `pipeline.py` calls
`predict_proba` with no model argument, so AutoGluon resolves `model_best`
internally.

Consequently **swapping the served model needs no code change** — export a clone
retaining a different model and `model_best` follows:

```bash
python trey_pipeline/ml_pipeline/export_deploy.py --model LightGBM_BAG_L1_FULL
```

### Ensemble vs LightGBM alone

Benchmarked on 5,850 human-adjudicated claims from
`unprocessed_claims_Matt_MCN_MARCH1Rev.csv` — a genuine holdout, since
`unprocessed_claims` is `verdict='U'` and disjoint by construction from the
`Y`/`N` rows used for training.

| | Ensemble | LightGBM only |
|---|---|---|
| balanced accuracy | 93.4% | 96.4% |
| false negatives | 46 | 24 |
| false Auto Yes | 5 | 18 |
| automation rate | 61.6% | 54.1% |
| size / predict time | 1.38 GB / 11.8s | 9 MB / 3.7s |

LightGBM looks safer at these thresholds, but that is an artifact of automating
less. Matching automation rate (`review = 0.20` instead of `0.1`) the two are
statistically indistinguishable: 20 vs 21 false negatives on 349 positives.

The ensemble is currently deployed. LightGBM is ~150× smaller and ~3× faster at
equivalent accuracy, so it is the likely future default.

## Caveats worth knowing

**The random-split holdout in `train.py` leaks.** 26.3% of its test rows share an
exact feature-vector with a training row — not duplicate claims, but *feature
collision*: nine coarse features over ~1,085 assets make distinct claims
indistinguishable to the model. Those rows score 0.009 log loss against 0.113
for the rest. Any metric from that split is optimistic. `train.py` has no
dedup and no grouped split.

Production input does **not** have this problem — 0 of 7,640 incoming claims
collide with training data — so the MARCH1Rev benchmark above is the trustworthy
one.

**Training uses only 80% of labelled data.** `refit_full` refits on the same 80%
passed to `fit`, not on everything; the remaining 20% is spent on evaluation and
threshold calibration.
