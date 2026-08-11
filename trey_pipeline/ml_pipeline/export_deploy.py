# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["autogluon.tabular[lightgbm,catboost]==1.5.0"]
# ///
"""Export an inference-only copy of the trained AutoGluon artifact.

The artifact produced by train.py carries everything AutoGluon needs to *keep
training*: unused candidate models, cached training data (utils/data/X.pkl,
~279 MB) and out-of-fold predictions. None of it is read at inference.

`clone_for_deployment` keeps only the best model and its dependencies, then
calls save_space() to drop the cached data. Predictions are unchanged -- see
verify_deploy.py, which asserts that byte-for-byte.

    python export_deploy.py                     # prune, keep the best model
    python export_deploy.py --model LightGBM_BAG_L1_FULL

The source artifact is never modified: clone_for_deployment writes to a new
path. Do not point --out at it.
"""
import argparse
import os
import shutil
from pathlib import Path

from autogluon.tabular import TabularPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = PROJECT_ROOT / "models" / "ag_challenger"
DEFAULT_OUT = PROJECT_ROOT / "models" / "ag_challenger_deploy"


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total


def mb(n):
    return f"{n / 1048576:,.1f} MB"


def export(src=DEFAULT_SRC, out=DEFAULT_OUT, model="best", overwrite=False):
    src, out = Path(src), Path(out)
    if src.resolve() == out.resolve():
        raise SystemExit("--out must differ from --src; the source is never modified")
    if out.exists():
        if not overwrite:
            raise SystemExit(f"{out} exists (pass --overwrite to replace)")
        shutil.rmtree(out)

    before = dir_size(src)
    print(f"source {src}  {mb(before)}")

    predictor = TabularPredictor.load(str(src))
    print(f"model_best = {predictor.model_best}")
    print(f"exporting  = {model}")

    predictor.clone_for_deployment(path=str(out), model=model, return_clone=False)

    after = dir_size(out)
    print(f"\nexported {out}  {mb(after)}")
    print(f"reclaimed {mb(before - after)}  ({100 * (before - after) / before:.1f}%)")

    kept = sorted(
        ((dir_size(p), p.name) for p in (out / "models").iterdir() if p.is_dir()),
        reverse=True,
    )
    print("\nmodels kept:")
    for size, name in kept:
        print(f"  {mb(size):>12}  {name}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC, help="trained artifact to read")
    ap.add_argument("--out", default=DEFAULT_OUT, help="where to write the inference-only copy")
    ap.add_argument("--model", default="best",
                    help="model to retain, e.g. LightGBM_BAG_L1_FULL (default: best)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    export(args.src, args.out, args.model, args.overwrite)
