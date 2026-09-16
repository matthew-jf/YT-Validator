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


## WESS language prediction

`predict_wess.py` predicts the WESS `language_id` (the number entered in the
monthly sheet, `WESS_LAN_num` in `sheets_language_families.csv`) for
unprocessed claims. Precision-first cascade — a tier only fires when its
cutoff, calibrated to >= 95% precision, is met; everything else is `REVIEW`:

1. `CHANNEL` — the channel's labeled history is unanimous (min claim count is tuned)
2. `TITLE` — the title contains a validated language-name rule. Names come from
   every name column of the sheet (`Anglicized_name`, `Language_JFProd`,
   `Language_name_WCD`, `Dialect_name`), plus native-name aliases mined from history,
   e.g. "bahasa melayu jambi". Phrases match by their *words*, not their word
   order, so "Creole Haitian" and "Haitian Creole" are one rule. When a title
   names several languages the longest name wins, and the last-named breaks a
   tie ("Creole French Haitian" -> Haitian, not French)
3. `FASTTEXT` — supervised fastText classifier over channel-prior tokens
   (channel's top historical languages, leave-one-out at fit time) + title text
3b. `ASR` — YouTube's automatic captions for the video, ISO -> WESS. A lookup
   costs 50 quota units, cannot be batched, and shares the 10,000/day key with
   production, so lookups are made by a daily collector and kept in a cache
   (see *Caption cache and daily collector*); training and prediction read the
   cache and spend nothing. ASR answers two kinds of row: **contested titles**
   (the title named several languages and the TITLE tiebreak had to choose — ASR
   may override it) and claims no cheaper tier answered. It is trusted **per
   language**: one ASR label often spans many WESS ids (`id` covers Djambi,
   Malaysian, North Moluccan Malay...), so certification measures each language
   separately — grouped by base code, `es-419` counts as `es` — and trusts only
   those reaching `ASR_PRECISION` on at least `ASR_MIN_PER_LANG` (25) rows. The
   training log also shows each language's base rate and lift, so a language that
   merely dominates the batch is visible. A failed lookup is never cached as "no
   captions", and a quota or key error stops the run. Measured on 150 fresh
   July held-out lookups (Sept 2026): 81% returned a track; `es` 27/27, `hi`
   20/22, `en` 10/11, `id` 4/41.
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
- `language_source` — `CHANNEL` / `TITLE` / `ASR` / `FASTTEXT` / `LID` / `REVIEW`
- `language_confidence` — 1.0 for exact-rule tiers, model probability otherwise

### Caption cache and daily collector

`data/asr_cache.jsonl` is an append-only log of every answered caption lookup:
`video_id`, the ASR language code (`''` when the video has no usable track) and
when it was fetched. It records what YouTube said, never whether to trust it, so
a language certified later can use answers fetched earlier.

Each `/predict` call writes its scored batch to `data/asr_queue.csv` and adds the
language columns above to the output (best effort: a language error never fails
the verdict run). The collector works through that batch once a day:

```shell
python predict_wess.py --collect-asr --prediction-input data/asr_queue.csv   # 180 lookups by default
```

It skips rows CHANNEL or TITLE already answer (except contested titles) and
anything cached, then looks up likely approvals first — `AUTO_Y`, then `REVIEW`,
then rows with no triage, then the near-certain N (`AUTO_N*` or a `licensed`
asset) — most-viewed first within each. `triage` and `licensed` are optional:
the daily ingest export carries neither, and then order is views alone. 180 lookups is 9,000
units, leaving room for the verdict pipeline's availability checks (~48 units per
run).

The workflow across both services:

```
DAILY    evidence only, nothing decided
  YouTube a3 report -> claims pipeline ingest (06:00 UTC) -> POST /asr/queue
    -> data/asr_queue.csv -> collector (08:15 UTC, 180 lookups) -> data/asr_cache.jsonl

MONTHLY  the decision, fed by that cache
  Ben's verdicts -> pipeline POST /predict + language_history + language_eval_labels
    -> certify ASR languages from the cache (zero quota) -> verdict model + language cascade
    -> CSV back to the pipeline -> Drive -> Ben

CONSOLE  browser cannot reach the VM, so the pipeline proxies
  console -> pipeline /api/claims-ingest/status -> localhost:3001/asr/status
```

Cadence and budget live in two different places, deliberately: the **cadence** is
the systemd timer (`OnCalendar=*-*-* 08:15:00 UTC`, after the Pacific quota
reset), and the **budget** is `ASR_DAILY_LIMIT` (180) in `predict_wess.py`, used
whenever `--asr-limit` is omitted. The limit is per invocation, not per calendar
day: a manual run spends on top of the timer's. Nothing can read YouTube's
remaining quota — the API does not expose it — so "9,000 of 10,000" is our own
arithmetic and assumes nothing else drew on the key.

Rates measured on the live queue (16 Sep 2026): claims arrive at ~110/day (107-125
depending on the window, from `claim_created_date`). How many need a lookup depends
on which tiers are live, because rows CHANNEL or TITLE answer are skipped. With the
deployed artifact (CHANNEL + FASTTEXT, TITLE self-disabled on the July batch)
99.5% need one, so ~110 lookups/day, net progress ~70/day against the 180 budget,
and the 4,405-video backlog clears in ~63 days. With TITLE live about 90% need one:
~99/day, ~54 days. Ben's verdicts remove claims from the queue before the collector
reaches them, so both are pessimistic bounds.

`GET /asr/status` is a cheap, pollable view for the claims console: queue rows
and whether the export carried `licensed`/`triage`, cache split (with and without
a caption track), the collector's last run (`looked_up`, `added`, `failed`,
`remaining`, `budget`, `stopped_reason` — `null`, `quota` or `outage`), live tiers
and trusted ASR languages, and branch/commit. It reads files only — line counts,
the collector's status JSON and the artifact re-parsed only when it changes — so
polling stays cheap as the cache grows. `remaining` is as of the last collector
run, not recomputed per request.

Monthly cycle: when the reviewed batch comes back, retrain with it as
`--eval-labels`. ASR certification reads the cache for the tuning half, and the
other half grades it.

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

### Daily ASR collector

```shell
sudo cp /opt/yt-validator/deploy/yt-validator-asr-collect.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now yt-validator-asr-collect.timer
systemctl list-timers yt-validator-asr-collect.timer
sudo journalctl -u yt-validator-asr-collect -n 50
```

Check the unit's `ExecStart` uses the same Python as `yt-validator`
(`systemctl cat yt-validator`). The collector does nothing until `/predict` has
written `data/asr_queue.csv`.
