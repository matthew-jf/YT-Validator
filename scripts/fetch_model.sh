#!/usr/bin/env bash
# Fetch the model artifact from GCS into trey_pipeline/models/.
#
# The artifact is ~1.4 GB, which cannot live in plain git (GitHub caps files at
# 100 MB and the RandomForest files are ~687 MB each). It is versioned by GCS
# path rather than tracked in the repo, so deployments pin a version instead of
# a commit.
#
#   MODEL_BUCKET=gs://my-bucket MODEL_VERSION=v1 ./scripts/fetch_model.sh
#
# Idempotent: an artifact already present and non-empty is left alone unless
# FORCE=1.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_BUCKET="${MODEL_BUCKET:-}"
MODEL_VERSION="${MODEL_VERSION:-v1}"
MODEL_NAME="${MODEL_NAME:-ag_challenger}"
DEST_ROOT="${MODEL_DIR:-$REPO_ROOT/trey_pipeline/models}"
DEST="$DEST_ROOT/$MODEL_NAME"
FORCE="${FORCE:-0}"

if [ -z "$MODEL_BUCKET" ]; then
  echo "ERROR: MODEL_BUCKET is not set (e.g. gs://jfp-yt-validator-models)" >&2
  exit 2
fi

SRC="${MODEL_BUCKET%/}/$MODEL_VERSION/$MODEL_NAME"

# "Present" must mean usable, not merely non-empty: a checkout without git-lfs
# leaves ~130-byte pointer stubs, which are non-empty and would otherwise be
# mistaken for a real artifact and skipped.
is_usable() {
  [ -s "$DEST/predictor.pkl" ] || return 1
  head -c 40 "$DEST/predictor.pkl" 2>/dev/null | grep -q "git-lfs.github.com" && return 1
  return 0
}

if [ "$FORCE" != "1" ] && is_usable; then
  echo "Model already present at $DEST (FORCE=1 to refetch)"
else
  if [ -e "$DEST/predictor.pkl" ] && ! is_usable; then
    echo "Existing artifact at $DEST is an unresolved LFS pointer; replacing it."
  fi
  echo "Fetching $SRC -> $DEST"
  mkdir -p "$DEST_ROOT"
  rm -rf "$DEST"
  # gcloud storage is faster than gsutil and present on modern images; fall back.
  if command -v gcloud >/dev/null 2>&1; then
    gcloud storage cp -r "$SRC" "$DEST_ROOT/"
  else
    gsutil -m cp -r "$SRC" "$DEST_ROOT/"
  fi
fi

# The threshold lives beside the artifact, not inside it, and belongs to the
# model version -- a different model implies a different calibrated threshold.
# Always refetch so a stale local copy cannot silently outrank the published
# one.
if true; then
  echo "Fetching ag_threshold.json"
  if command -v gcloud >/dev/null 2>&1; then
    gcloud storage cp "${MODEL_BUCKET%/}/$MODEL_VERSION/ag_threshold.json" "$DEST_ROOT/"
  else
    gsutil cp "${MODEL_BUCKET%/}/$MODEL_VERSION/ag_threshold.json" "$DEST_ROOT/"
  fi
fi

# Guard against the failure that motivated this script: unresolved Git LFS
# pointers are ~130-byte text files that load as garbage, and the service used
# to report healthy while every prediction failed.
echo "Verifying artifact..."
bad=0
while IFS= read -r -d '' f; do
  if head -c 40 "$f" 2>/dev/null | grep -q "git-lfs.github.com"; then
    echo "  UNRESOLVED LFS POINTER: $f" >&2
    bad=1
  fi
done < <(find "$DEST" -name '*.pkl' -print0)

for required in "$DEST/predictor.pkl" "$DEST/learner.pkl" "$DEST_ROOT/ag_threshold.json"; do
  if [ ! -s "$required" ]; then
    echo "  MISSING OR EMPTY: $required" >&2
    bad=1
  fi
done

if [ "$bad" != "0" ]; then
  echo "Artifact verification FAILED" >&2
  exit 1
fi

echo "OK: $(du -sh "$DEST" | cut -f1) at $DEST"
