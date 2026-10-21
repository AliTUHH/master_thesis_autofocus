#!/bin/bash
#SBATCH --job-name=holofocus-predict
#SBATCH --nodes=1
#SBATCH --time=01:00:00
# Learned estimates for the samples of a split, e.g. as the warm start of the baseline:
#
#   submit.sh predict MODEL DATA NAME
#
# MODEL is a training run directory (committed, e.g. results/learned/spectrum-cnn-wide, or
# under $HOLOFOCUS_SCRATCH/runs), DATA a split file and NAME the output directory under
# $HOLOFOCUS_SCRATCH/runs. If MODEL holds predictions for a split of the same name, the new
# ones are compared with them sample by sample.
set -euo pipefail
# shellcheck source=SCRIPTDIR/../lib.sh
source "${HOLOFOCUS_KIT:?submit through cluster/maxwell/submit.sh}/lib.sh"
kit_job_start

usage="usage: predict MODEL DATA NAME"
model=${1:?$usage}
data=$(kit_data "${2:?$usage}")
out=$(kit_run_dir "${3:?$usage}")
kit_do holofocus-learn predict "$model" --data "$data" --out "$out" --device "$HOLOFOCUS_DEVICE"
status=0
split=$(python -c 'import sys, h5py; print(h5py.File(sys.argv[1], "r").attrs["split"])' "$data")
if [[ -f $model/$split/predictions.csv ]]; then
    kit_do python scripts/check_cluster_run.py predictions "$out/predictions.csv" \
        --reference "$model/$split/predictions.csv" || status=1
fi
kit_keep "$out"
exit "$status"
