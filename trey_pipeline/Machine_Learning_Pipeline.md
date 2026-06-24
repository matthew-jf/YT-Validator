# AI Claims Determination Pipeline

## Overview
This repository contains the end-to-end automated machine learning pipeline for the AI Claims Determination System. The system evaluates historical and incoming YouTube copyright claims, using metadata relationships and textual similarity to assign one of three operational decisions:
* **Auto Yes:** High-confidence approval.
* **Auto No:** High-confidence rejection.
* **Human Review:** Edge cases requiring manual intervention.

The pipeline is designed to be fully modular, reproducible, and executable on a local machine without requiring complex virtual environment setups, thanks to the use of `uv` for inline dependency management.

---

## Architecture & File Structure
The project is broken down into distinct, logical components to separate feature engineering, training, inference, and visualization. 

* **`feature_utils.py`**: The shared engine. It houses the data cleaning, duration mathematics, and fuzzy string-matching logic (via `thefuzz`). Centralizing this ensures that the exact same transformations are applied to both historical training data and new, unseen inference data.
* **`train.py`**: The model generation script. It ingests historical data, trains the baseline and challenger models, calibrates the safety thresholds, and outputs the artifacts to the `models/` directory.
* **`inference.py`**: The operational script. It loads the pre-trained models and applies them to new, incoming monthly CSVs without needing to retrain.
* **`visualize.py`**: The performance scorecard generator. If ground truth data is available, it calculates automation rates, accuracy, and false negatives, outputting a Seaborn chart (`workload_reduction.png`).
* **`pipeline.sh`**: The master orchestrator. A single bash script that checks for existing models, trains them if missing, runs inference, and generates visualizations automatically.

---

## The Dual-Model Strategy

To balance speed, interpretability, and predictive power, this system utilizes a two-tier "Baseline vs. Challenger" architecture.

### 1. The Baseline: XGBoost (`xgb_baseline.json`)
* **Role:** Baseline model for initial testing. 
* **Design:** A lightweight gradient boosting classifier trained on a subset of mathematical and structural features (duration diffs, fuzzy text ratios). It is explicitly weighted to be overly harsh on False Negatives to protect against bad predictions.

### 2. The Challenger: AutoGluon (`ag_challenger/`)
* **Role:** Secondary ensemble stack. Primary decision maker.
* **Design:** AutoGluon automatically trains and ensembles multiple model types (LightGBM, CatBoost, Random Forests, etc.). 

---

## How to Execute the Pipeline

Everything is containerized via PEP 723 inline dependencies and the `uv` package manager. You do not need to manually configure a `requirements.txt` or `venv`.

### First-Time Setup
Ensure the bash script is executable:
```bash
chmod +x pipeline.sh
```
or open a bash terminal and run:
```bash
cd trey_pipeline


./pipeline.sh
```