
# conda env update -n YT-Validator -f environment.yml
"""Does the train/test split leak?

python check_split_leak.py [claims.csv]

Counts test rows whose 9 features are identical to a training row. The model
cannot tell those apart, so it answers them from memory and the score flatters
it. Compares the old row-wise split against grouping.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from feature_utils import engineer_features, AG_FEATURES


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
data = sys.argv[1] if len(sys.argv) > 1 else PIPELINE_ROOT / "data" / "all_claims.csv"

df = pd.read_csv(data, engine="python", on_bad_lines="skip", encoding="utf-8-sig")
df.columns = df.columns.astype(str).str.strip()
df = df[df.verdict.isin(["Y", "N"])]
y = df.verdict.map({"Y": 1, "N": 0})

# one string per row: its 9 features joined. identical strings = identical claims
key = engineer_features(df)[AG_FEATURES].astype(str).agg("|".join, axis=1)
print(f"{len(df):,} claims, {key.nunique():,} distinct feature-vectors\n")


def leak(train_idx, test_idx, label):
    seen = set(key.iloc[train_idx])
    n = int(key.iloc[test_idx].isin(seen).sum())
    print(f"  {label:<22} {n:>7,} of {len(test_idx):,} test rows already seen "
          f"({100 * n / len(test_idx):.1f}%)")


leak(*train_test_split(range(len(df)), test_size=0.2, random_state=42, stratify=y), "train_test_split")