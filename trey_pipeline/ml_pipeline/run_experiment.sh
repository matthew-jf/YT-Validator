#!/usr/bin/env bash
# Train Trey's model on one claims file, grade it on another against Ben's verdicts.
#
#   run_experiment.sh <name> <train_all_claims.csv> <test_unprocessed.csv> <test_verdicts.csv>
#
# Trains into models/experiments/<name>/ (never touches models/ag_challenger),
# then grades. Everything printed is also saved to <dir>/report.txt.
set -euo pipefail
[ $# -eq 4 ] || { sed -n '2,5p' "$0"; exit 2; }

name=$1; train=$2; claims=$3; verdicts=$4
here=$(cd "$(dirname "$0")" && pwd)
out="$here/../models/experiments/$name"
PY=${PY:-/opt/anaconda3/envs/YT-Validator/bin/python}

[ -e "$out/ag_challenger/predictor.pkl" ] && { echo "$out already trained — pick a new name or delete it"; exit 1; }
mkdir -p "$out"

{
  echo "experiment : $name"
  echo "git        : $(git -C "$here" rev-parse --short HEAD) ($(git -C "$here" branch --show-current))"
  echo "train      : $train   ($(( $(wc -l < "$train") - 1 )) rows)"
  echo "test       : $claims"
  echo "verdicts   : $verdicts"
  echo "started    : $(date '+%Y-%m-%d %H:%M')"
  echo
  echo "=== train ==="
  (cd "$here" && $PY -c "from train import run_training; run_training('$train', '$out')")
  echo
  echo "=== grade ==="
  MODEL_DIR="$out" $PY "$here/grade_against_verdicts.py" "$claims" "$verdicts"
  echo
  echo "finished   : $(date '+%Y-%m-%d %H:%M')"
} 2>&1 | grep -vE "mismatch|INFO: AutoGluon|WARNING: System" | tee "$out/report.txt"
