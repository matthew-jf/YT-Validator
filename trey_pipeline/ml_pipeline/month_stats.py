
# conda env update -n YT-Validator -f environment.yml
"""How well does the model rank one month's claims, and how fresh is it?

python month_stats.py <claims.csv> <verdicts.csv>

AUC is threshold-free: the chance a valid claim outranks an invalid one.
Unlike the miss rate it doesn't move with the class mix, so it isolates the
model's ranking from where the thresholds happen to sit.
"""
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score
from autogluon.tabular import TabularPredictor
from feature_utils import engineer_features, AG_FEATURES
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"


def read_csv(path):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip", encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()
    return df


def training_cutoff():
    """Newest claim the model was trained on."""
    d = read_csv(PROJECT_ROOT / "data" / "all_claims.csv")
    return pd.to_datetime(d.claim_created_date, errors="coerce", format="mixed").max()


def main(claims_path, verdicts_path):
    claims = read_csv(claims_path)
    verdicts = read_csv(verdicts_path)[["video_id", "verdict"]].dropna(subset=["video_id"])

    rating = (TabularPredictor.load(str(MODEL_DIR / "ag_challenger"))
              .predict_proba(engineer_features(claims)[AG_FEATURES]).iloc[:, 1])
    scored = claims[["video_id"]].assign(rating=rating.to_numpy())

    graded = scored.merge(verdicts[verdicts.verdict.isin(["Y", "N"])], on="video_id")
    valid = graded.verdict == "Y"

    created = pd.to_datetime(claims.claim_created_date, errors="coerce", format="mixed").median()
    age = (created - training_cutoff()).days

    print(f"{Path(claims_path).parent.name:<10} {len(graded):>6,} graded  "
          f"median claim {created.date()} ({age:>4}d past training)  "
          f"valid {100 * valid.mean():>2.0f}%  AUC {roc_auc_score(valid, graded.rating):.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
