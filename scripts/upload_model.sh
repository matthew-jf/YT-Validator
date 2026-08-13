#!/usr/bin/env bash
# Publish a model artifact to GCS under an immutable version prefix.
#
# Run this once per trained model, after export_deploy.py has pruned it and
# verify_deploy.py has confirmed the pruning is lossless:
#
#   python trey_pipeline/ml_pipeline/export_deploy.py
#   python trey_pipeline/ml_pipeline/verify_deploy.py
#   MODEL_BUCKET=gs://my-bucket MODEL_VERSION=v1 ./scripts/upload_model.sh
#
# Versions are immutable by convention: to ship a new model, bump
# MODEL_VERSION rather than overwriting. Deployments then pin a version, and
# rolling back is a config change instead of a rebuild.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_BUCKET="${MODEL_BUCKET:-}"
MODEL_VERSION="${MODEL_VERSION:-}"
SRC_NAME="${SRC_NAME:-ag_challenger_deploy}"
DEST_NAME="${DEST_NAME:-ag_challenger}"
MODELS_DIR="$REPO_ROOT/trey_pipeline/models"
SRC="$MODELS_DIR/$SRC_NAME"

if [ -z "$MODEL_BUCKET" ] || [ -z "$MODEL_VERSION" ]; then
  echo "Usage: MODEL_BUCKET=gs://bucket MODEL_VERSION=v1 $0" >&2
  exit 2
fi
if [ ! -s "$SRC/predictor.pkl" ]; then
  echo "ERROR: no artifact at $SRC -- run export_deploy.py first" >&2
  exit 1
fi

DEST="${MODEL_BUCKET%/}/$MODEL_VERSION"

if command -v gcloud >/dev/null 2>&1; then
  if gcloud storage ls "$DEST/$DEST_NAME/predictor.pkl" >/dev/null 2>&1; then
    echo "ERROR: $DEST/$DEST_NAME already exists; bump MODEL_VERSION" >&2
    exit 1
  fi
  gcloud storage cp -r "$SRC" "$DEST/$DEST_NAME"
  gcloud storage cp "$MODELS_DIR/ag_threshold.json" "$DEST/ag_threshold.json"
else
  if gsutil -q stat "$DEST/$DEST_NAME/predictor.pkl" 2>/dev/null; then
    echo "ERROR: $DEST/$DEST_NAME already exists; bump MODEL_VERSION" >&2
    exit 1
  fi
  gsutil -m cp -r "$SRC" "$DEST/$DEST_NAME"
  gsutil cp "$MODELS_DIR/ag_threshold.json" "$DEST/ag_threshold.json"
fi

echo "Published $(du -sh "$SRC" | cut -f1) to $DEST/$DEST_NAME"
echo "Deploy with: MODEL_BUCKET=$MODEL_BUCKET MODEL_VERSION=$MODEL_VERSION"
