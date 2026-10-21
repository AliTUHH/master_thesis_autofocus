#!/bin/bash
#SBATCH --job-name=holofocus-gpu-check
#SBATCH --nodes=1
#SBATCH --time=01:00:00
# Check a GPU node before the long runs (docs/maxwell-gpu.md, section 4):
#
#   submit.sh gpu-check
#
# 1. PyTorch and CUDA on the device: FFT, convolution, the forward model and one ASRM
#    reconstruction against the CPU, and the time per reconstruction at padding 2 and 4.
# 2. The first 4 samples of the cpu-4x test split, generated on the device: labels and
#    holograms against results/datasets/cpu-4x-test.csv, the committed main model on them
#    against its committed predictions, and the classical autofocus of sample 0 against
#    Stage 3 (results/baseline/cpu-4x-test).
# Every check prints PASS, WARN or FAIL; the job fails if a check fails.
set -euo pipefail
# shellcheck source=SCRIPTDIR/../lib.sh
source "${HOLOFOCUS_KIT:?submit through cluster/maxwell/submit.sh}/lib.sh"
kit_job_start

out=$(kit_run_dir "gpu-check/${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}")
mkdir -p "$out"
check=(python scripts/check_cluster_run.py)
status=0
kit_do "${check[@]}" device --device "$HOLOFOCUS_DEVICE" --out "$out/device.json" || status=1
kit_do holofocus-dataset generate configs/dataset/cpu-4x.yaml --splits test --limit 4 \
    --device "$HOLOFOCUS_DEVICE" --out "$out/data"
kit_do "${check[@]}" data "$out/data/test.h5" --reference results/datasets/cpu-4x-test.csv ||
    status=1
kit_do holofocus-learn predict results/learned/spectrum-cnn-wide --data "$out/data/test.h5" \
    --out "$out/predict" --device "$HOLOFOCUS_DEVICE"
kit_do "${check[@]}" predictions "$out/predict/predictions.csv" \
    --reference results/learned/spectrum-cnn-wide/test/predictions.csv || status=1
kit_do holofocus-baseline run configs/baseline/paper2-cpu.yaml --data "$out/data/test.h5" \
    --out "$out/baseline" --indices 0 --device "$HOLOFOCUS_DEVICE"
kit_do "${check[@]}" baseline "$out/baseline" --reference results/baseline/cpu-4x-test ||
    status=1
kit_keep "$out"
if ((status)); then
    echo "== gpu-check: a check FAILED (see above)"
else
    echo "== gpu-check: no check failed"
fi
exit "$status"
