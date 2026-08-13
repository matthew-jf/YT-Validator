# Deploy runbook

YT-Validator runs on the `ytdt-claims` GCE VM under systemd, alongside the Node
pipeline, reachable at `http://localhost:3001`. The VM itself is provisioned by
[ytdt-claims-pipeline](https://github.com/JesusFilm/ytdt-claims-pipeline/blob/main/docs/deploy.md);
this document covers the ML service and its model artifact.

The model is **not in git**. It lives in GCS and is fetched by
[`scripts/fetch_model.sh`](../scripts/fetch_model.sh). See [model.md](model.md)
for why and how it is produced.

---

## Publish a model version

Do this once per trained model, before any deploy that needs it.

```bash
python trey_pipeline/ml_pipeline/export_deploy.py     # prune, ~3.1 GB -> ~1.4 GB
python trey_pipeline/ml_pipeline/verify_deploy.py     # must print PASS
MODEL_BUCKET=gs://jfp-yt-validator-models MODEL_VERSION=v1 ./scripts/upload_model.sh
```

`verify_deploy.py` asserts the pruned artifact predicts identically to the
source. If it does not print `PASS`, stop — do not publish.

Versions are immutable: `upload_model.sh` refuses to overwrite. To ship a new
model, bump `MODEL_VERSION`. Rollback is then a metadata change, not a rebuild.

---

## Update the service on a running VM

The VM already exists, so `deploy.sh` will not re-run provisioning
(`cloud-config.yaml` is user-data and only executes on first boot). Update in
place.

**1. Pull the code.**

```bash
gcloud compute ssh ytdt-claims --zone=us-east1-b --command="sudo git -C /opt/yt-validator fetch origin && sudo git -C /opt/yt-validator checkout -f -B ML-Pipeline origin/ML-Pipeline"
```

`checkout -f` discards local modifications. `.env` and `data/` are gitignored
and survive. Substitute any branch for `ML-Pipeline` when testing before merge.

**2. Update the conda env** — only when `environment.yml` changed. Takes ~10 min.

```bash
gcloud compute ssh ytdt-claims --zone=us-east1-b --command="cd /opt/yt-validator && sudo /opt/miniconda3/bin/conda env update -n YT-Validator -f environment.yml"
```

**3. Fetch the model.**

```bash
gcloud compute ssh ytdt-claims --zone=us-east1-b --command="sudo MODEL_BUCKET=gs://jfp-yt-validator-models MODEL_VERSION=v1 bash /opt/yt-validator/scripts/fetch_model.sh"
```

Idempotent — skips when a usable artifact is already present. `FORCE=1` refetches.

**4. Restart and verify.**

```bash
gcloud compute ssh ytdt-claims --zone=us-east1-b --command="sudo systemctl restart yt-validator && sleep 60 && curl -s localhost:3001/health"
```

Expect `"status": "healthy"` with `"warm": true`. **The model is loaded and
warmed before Flask binds its port**, so an immediate `curl` returns nothing —
expected, not a failure. Budget ~60s after a cold boot (~30s to read 1.4 GB from
disk) and ~10s when the page cache is still warm, e.g. right after a refetch.

---

## Rollback

```bash
gcloud compute ssh ytdt-claims --zone=us-east1-b --command="sudo git -C /opt/yt-validator checkout -f <previous-ref> && sudo FORCE=1 MODEL_BUCKET=gs://jfp-yt-validator-models MODEL_VERSION=v1 bash /opt/yt-validator/scripts/fetch_model.sh && sudo systemctl restart yt-validator"
```

To roll back only the *model*, change `MODEL_VERSION` and refetch with `FORCE=1`
— no code change needed.

`FORCE=1` matters when rolling back to a ref that predates the model leaving
git: those refs still track the artifact, so the checkout writes Git LFS pointer
stubs over the real files.

---

## Verifying a deploy

`/health` returns **200 only when the model is loaded and warm**, and **503**
otherwise. It is a readiness signal, not a liveness stub.

```bash
curl -s localhost:3001/health
```

```json
{
  "branch": "ML-Pipeline", "commit": "0f5ee69", "status": "healthy",
  "model": {
    "model_best": "WeightedEnsemble_L2_FULL", "loaded": true, "warm": true,
    "warm_seconds": 30.21, "peak_rss_mb": 1632.1, "load_seconds": 0.26,
    "auto_yes_threshold": 0.97, "review_threshold": 0.1,
    "model_dir": "/opt/yt-validator/trey_pipeline/models/ag_challenger"
  },
  "running_tasks": 0
}
```

Check `branch`/`commit` to confirm which code is live, and `model_best` to
confirm which model is serving.

`/health` proves the model loads. It does not prove scoring works — for that,
run a real prediction and inspect the `telemetry` block on the task record.

### Expected footprint

Measured on `e2-standard-4` (4 vCPU / 16 GB) with the pruned ensemble:

| | |
|---|---|
| peak RSS | ~1,630 MB |
| model warm-up | ~30s cold, ~6s with warm page cache |
| predict, 105K rows | ~12s |
| artifact on disk | ~1.4 GB |

The model is loaded **once per process**, not per request. Runs are serialised
by a semaphore: one scoring job saturates the available cores, and concurrent
jobs would only multiply peak memory.

---

## Troubleshooting

### Service is `active` but nothing answers on 3001

Either still warming (~55s from restart) or crash-looping. `Restart=always`
makes a crash look like a running service.

```bash
gcloud compute ssh ytdt-claims --zone=us-east1-b --command="systemctl show yt-validator -p NRestarts --value && sudo journalctl -u yt-validator -n 50 --no-pager"
```

A climbing restart counter means it is failing at startup, and the traceback
will be in the log.

### `TypeError: NDFrame.fillna() got an unexpected keyword argument 'downcast'`

pandas 3.0 removed an argument AutoGluon 1.5.0 still calls. `environment.yml`
bounds pandas and numpy to AutoGluon's own ranges to prevent this; the failure
means something installed out-of-range versions.

```bash
gcloud compute ssh ytdt-claims --zone=us-east1-b --command="/opt/miniconda3/envs/YT-Validator/bin/python -c 'import pandas, numpy; print(pandas.__version__, numpy.__version__)'"
```

Expect pandas `2.x` (`<2.4`) and numpy `<2.4`. If `conda list` and the runtime
disagree, the env has duplicate `dist-info` directories — pip reads stale
metadata and silently refuses to correct the version. Clear and reinstall:

```bash
SP=/opt/miniconda3/envs/YT-Validator/lib/python3.11/site-packages
sudo $SP/../../../bin/python -m pip uninstall -y pandas numpy
sudo rm -rf $SP/pandas $SP/pandas-*.dist-info $SP/numpy $SP/numpy-*.dist-info
sudo $SP/../../../bin/python -m pip install 'pandas>=2.0.0,<2.4.0' 'numpy>=1.25.0,<2.4.0'
```

### `Artifact verification FAILED — UNRESOLVED LFS POINTER`

A checkout on a host without `git-lfs` wrote ~130-byte pointer stubs where the
model should be. `fetch_model.sh` detects this and refuses rather than letting
the service start broken. Refetch:

```bash
sudo FORCE=1 MODEL_BUCKET=gs://jfp-yt-validator-models MODEL_VERSION=v1 bash /opt/yt-validator/scripts/fetch_model.sh
```

This cannot happen on refs where the artifact is untracked; it only affects
older ones.

### `/health` returns 503

The model did not load. Common causes: artifact missing or stubbed (above),
`MODEL_NAME` pointing at a directory that does not exist, or an env problem.
The reason is always in `journalctl -u yt-validator -n 50`.

### Tasks stuck `running`

A crash mid-run used to leave tasks marked `running` forever, so the webhook
never fired and the caller waited indefinitely. Orphans are now failed at
startup — check the log for `Reconciled N orphaned task(s)`.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `YT_API_KEY` | — | required; YouTube Data API, 1 unit per 50 videos |
| `MODEL_NAME` | `ag_challenger` | artifact directory under `MODEL_DIR` |
| `MODEL_DIR` | `trey_pipeline/models` | where artifacts live |
| `MODEL_BUCKET` | — | GCS bucket for `fetch_model.sh` / `upload_model.sh` |
| `MODEL_VERSION` | `v1` | version prefix in the bucket |
| `WARM_MODEL` | `1` | `0` skips startup load (tests, CLI use) |
| `FLASK_RUN_PORT` | `3001` | |

On the VM these come from the systemd unit and `/opt/yt-validator/.env`, written
by `ExecStartPre`.

---

## Retrieving results

Output CSVs are written to `data/output_<task_id>.csv` on the service host and
served by `GET /download/<task_id>`. The service listens on localhost only, so
pull them off the VM with `scp`:

```bash
gcloud compute scp ytdt-claims:/opt/yt-validator/data/output_<task_id>.csv ./data/ --zone=us-east1-b
```

List what is available:

```bash
gcloud compute ssh ytdt-claims --zone=us-east1-b --command="ls -lh /opt/yt-validator/data/output_*.csv"
```

The output is the input CSV plus `rating`, `action`, `licensed`,
`media_component_id` and `video_available`. In normal operation the Node
pipeline fetches it via the webhook's `csv_path` and uploads it to Drive; this
is for spot checks.
