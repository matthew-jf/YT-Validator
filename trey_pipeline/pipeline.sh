#!/usr/bin/env bash
set -e
export UV_LINK_MODE=copy

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Ensure directories exist
mkdir -p models
mkdir -p data

# 1. Pipeline Check: Do all required model artifacts exist?
if [ ! -f "models/xgb_baseline.json" ] || [ ! -d "models/ag_challenger" ] || [ ! -f "models/ag_threshold.json" ]; then
    echo "One or more model artifacts are missing. Initiating training..."
    uv run --python 3.11 ml_pipeline/train.py
else
    echo "Pre-trained model artifacts found. Skipping training phase."
fi

# 2. Run Inference on new data
if [ -f "data/output_claims.csv" ]; then
    echo "Inference output ('data/output_claims.csv') already exists. Skipping inference step."
else
    echo "Executing Inference..."
    uv run --python 3.11 ml_pipeline/inference.py
fi

# # 3. (WIP) Mapper enrichment for Auto Yes claims
# echo "Generating MCID mapper guesses for Auto Yes claims..."
# uv run --python 3.11 mcid_mapper/mcid_mapper.py

# 4. Language Checker for Auto Yes claims
if [ -f "data/language_evaluation_report.csv" ]; then
    echo "Language evaluation report ('data/language_evaluation_report.csv') already exists. Skipping language check step."
else
    echo "Generating Language guesses for Auto Yes claims..."
    uv run --python 3.11 language_checker/google_and_whisper.py
fi

# 5. Validation / Visualization
echo "Generating Performance Scorecards..."
uv run --python 3.11 ml_pipeline/visualize.py