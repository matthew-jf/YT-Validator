
# conda env update -n YT-Validator -f environment.yml
"""Grade Trey's model against Ben's verdicts.

python grade_against_verdicts.py <claims.csv> <verdicts.csv>
    <claims.csv>   an unprocessed-claims export (from Drive or the pipeline)
    <verdicts.csv> the reviewed file: video_id + verdict (Y/N)

Scores the claims, applies the model-driven decision rules, then joins on
video_id and reports how automated decisions compares to Ben's.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import balanced_accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_utils import engineer_features, AG_FEATURES

MODEL_DIR = Path(os.environ.get("MODEL_DIR", PROJECT_ROOT / "models"))
AUTO_YES = 0.97
REVIEW = json.load(open(MODEL_DIR / "ag_threshold.json"))["threshold"]


def read_csv(path):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip", encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip().str.replace("﻿", "")
    return df


def score(claims):
    """Model probability, and the duration difference the rules need."""
    features = engineer_features(claims)
    predictor = TabularPredictor.load(str(MODEL_DIR / "ag_challenger"))
    rating = predictor.predict_proba(features[AG_FEATURES]).iloc[:, 1].to_numpy()
    return rating, features.duration_diff_sec.to_numpy()


def decide(rating, duration_diff):
    """pipeline.py's three-way decision."""
    action = np.where(rating >= AUTO_YES, "Auto Yes", np.where(rating >= REVIEW, "Human Review", "Auto No"))
    action[(action == "Auto No") & (duration_diff == 0)] = "Human Review"  # exact match, look again
    return action


def main(claims_path, verdicts_path):
    claims = read_csv(claims_path)
    rating, duration_diff = score(claims)
    claims["action"] = decide(rating, duration_diff)
    claims["rating"] = rating

    verdicts = read_csv(verdicts_path)[["video_id", "verdict"]].dropna(subset=["video_id"])
    verdicts = verdicts[verdicts.verdict.isin(["Y", "N"])]
    graded = claims.drop(columns=["verdict"], errors="ignore").merge(verdicts, on="video_id")   # trust only Ben's verdict, not the export's
    valid = graded.verdict == "Y"                                                               # Ben said the claim is good
    automated = graded.action != "Human Review"         

    missed = int((valid & (graded.action == "Auto No")).sum())
    wrongly_approved = int((~valid & (graded.action == "Auto Yes")).sum())
    balanced = balanced_accuracy_score(valid[automated], graded.action[automated] == "Auto Yes")

    print(f"{len(graded):,} claims with a human verdict ({int(valid.sum()):,} valid, {100 * valid.mean():.0f}%)\n")
    print(f"  automated            {100 * automated.mean():5.1f}%  ({int(automated.sum()):,} decided without a person)")
    print(f"  balanced accuracy    {100 * balanced:5.1f}%\n")
    print(f"  MISSED valid claims  {missed:5,}  of {int(valid.sum()):,} ({100 * missed / valid.sum():.0f}% auto-rejected in error)")
    print(f"  wrongly approved     {wrongly_approved:5,}\n")

    for name in ["Auto Yes", "Human Review", "Auto No"]:
        n = int((graded.action == name).sum())
        print(f"  {name:<14} {n:6,}  ({100 * n / len(graded):4.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])

