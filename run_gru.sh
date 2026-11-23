#!/usr/bin/env bash

set -o pipefail

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
LOGS=logs_gru
mkdir -p "$LOGS"
EXTRA=""
[ "${QUICK:-0}" = "1" ] && EXTRA="--quick"
DEV="--device ${DEVICE:-auto}"

pack() {
  {
    echo "date: $(date)"
    echo "os: $(uname -a)"
    python -c "import sys, numpy, pandas; print('python', sys.version.split()[0]); print('numpy', numpy.__version__); print('pandas', pandas.__version__)"
    python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())" 2>&1
    echo "git branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
    git status --short 2>/dev/null
  } > "$LOGS/environment.txt" 2>&1
  rm -f gru_results.zip
  zip -qr gru_results.zip "$LOGS" results/deep report/tables/gru_*.tex \
      -x "*.npz" 2>/dev/null
  echo
  echo ">>> Results and logs packed in $(pwd)/gru_results.zip"
}

step() {
  name=$1; shift
  echo
  echo "=================== $name: $*"
  "$@" 2>&1 | tee "$LOGS/$name.log"
  code=${PIPESTATUS[0]}
  if [ "$code" -ne 0 ]; then
    echo "!!! FAILED at step '$name' (exit code $code). Full output: $LOGS/$name.log"
    pack
    exit 1
  fi
}

step 0_imports python -c "import eventflow.train_deep, torch; print('imports ok, torch', torch.__version__)"
step 1_tests   python -m pytest tests -q
step 2_smoke   python -m eventflow.train_deep smoke
AWAKE=""
command -v caffeinate >/dev/null 2>&1 && AWAKE="caffeinate -i"   # Mac: don't sleep mid-run
step 3_tune    $AWAKE python -m eventflow.train_deep tune --resume $EXTRA $DEV
step 4_ablate  $AWAKE python -m eventflow.train_deep ablate --resume $DEV
step 5_compare python -m eventflow.train_deep compare
if [ -f scripts/deep_report_assets.py ]; then
  step 6_report_assets python scripts/deep_report_assets.py
fi
echo
echo "All steps finished."
pack
