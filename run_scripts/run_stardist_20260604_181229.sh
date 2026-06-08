#!/usr/bin/env bash
set -euo pipefail
cd "/home/ksc/projects/stardistTest"
echo "===== StarDist run started: $(date) ====="
echo "PROJECT=/home/ksc/projects/stardistTest"
echo "KERNEL=dsb-stardist-cuda"
echo "OUT_NOTEBOOK=/home/ksc/projects/stardistTest/executed_notebooks/stardist_dsb2018_train_executed_20260604_181229.ipynb"
echo "LOG=/home/ksc/projects/stardistTest/logs/stardist_run_20260604_181229.log"
echo "nvidia-smi:"
nvidia-smi || true
echo "python gpu check:"
"/home/ksc/projects/stardistTest/.venv/bin/python" - <<'PY'
import tensorflow as tf
print('TF', tf.__version__)
print('GPUS', tf.config.list_physical_devices('GPU'))
PY
echo "===== nbconvert execute ====="
"/home/ksc/projects/stardistTest/.venv/bin/python" -m jupyter nbconvert   --to notebook   --execute "/home/ksc/projects/stardistTest/stardist_dsb2018_train.ipynb"   --ExecutePreprocessor.kernel_name=dsb-stardist-cuda   --ExecutePreprocessor.timeout=-1   --output "/home/ksc/projects/stardistTest/executed_notebooks/stardist_dsb2018_train_executed_20260604_181229.ipynb"
echo "===== StarDist run finished: $(date) ====="
