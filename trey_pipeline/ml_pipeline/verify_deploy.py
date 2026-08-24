# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["autogluon.tabular[lightgbm,catboost]==1.5.0", "pandas", "numpy", "thefuzz"]
# ///
"""Assert a pruned artifact predicts identically to the one it came from.

Pruning is only safe if it is lossless. This scores the same rows through both
artifacts and compares the raw probabilities, then the three-way actions from
pipeline.py. Exits non-zero on any difference, so it can gate a build.

    python verify_deploy.py                       # uses unprocessed_claims.csv
    python verify_deploy.py --data ../data/all_claims.csv --limit 50000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from feature_utils import engineer_features, AG_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTO_YES_THRESHOLD = 0.97


def load_claims(path, limit=None):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip", encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()
    return df.head(limit) if limit else df


def actions(rating, duration_diff, review_threshold):
    a = np.where(rating >= AUTO_YES_THRESHOLD, "Auto Yes",
                 np.where(rating >= review_threshold, "Human Review", "Auto No"))
    a[(a == "Auto No") & (duration_diff == 0)] = "Human Review"
    return a


def main(src, out, data, limit):
    review_threshold = json.load(open(PROJECT_ROOT / "models" / "ag_threshold.json"))["threshold"]
    df = load_claims(data, limit)
    features = engineer_features(df)
    X = features[AG_FEATURES]
    duration_diff = features["duration_diff_sec"].to_numpy()
    print(f"scoring {len(df):,} rows from {Path(data).name}")

    scores = {}
    for label, path in (("source", src), ("pruned", out)):
        predictor = TabularPredictor.load(str(path))
        scores[label] = predictor.predict_proba(X).iloc[:, 1].to_numpy()
        print(f"  {label:<7} {path}  model_best={predictor.model_best}")

    a, b = scores["source"], scores["pruned"]
    identical = np.array_equal(a, b)
    max_delta = float(np.max(np.abs(a - b))) if len(a) else 0.0

    act_a = actions(a, duration_diff, review_threshold)
    act_b = actions(b, duration_diff, review_threshold)
    action_diffs = int((act_a != act_b).sum())

    print(f"\nmax probability delta : {max_delta:.3e}")
    print(f"actions differing     : {action_diffs:,} / {len(df):,}")

    if identical and action_diffs == 0:
        print("\nPASS - pruned artifact is bit-identical")
        return 0
    print("\nFAIL - pruned artifact diverges from source")
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=PROJECT_ROOT / "models" / "ag_challenger")
    ap.add_argument("--out", default=PROJECT_ROOT / "models" / "ag_challenger_deploy")
    ap.add_argument("--data", default=PROJECT_ROOT / "data" / "unprocessed_claims.csv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    sys.exit(main(args.src, args.out, args.data, args.limit))
