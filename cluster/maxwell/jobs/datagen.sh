#!/bin/bash
#SBATCH --job-name=holofocus-datagen
#SBATCH --nodes=1
#SBATCH --time=04:00:00
# Generate the splits of a dataset config into $HOLOFOCUS_SCRATCH/data/<name of the config>:
#
#   submit.sh [--cpu] datagen CONFIG [--splits SPLIT ...] [--limit N]
#
# CONFIG is a file of configs/dataset/. Labels and random streams do not depend on the
# machine, the noise-free intensity does at float32 rounding, and with it a few noise values
# of some samples. For val and test splits that should match the committed results, use
# --cpu (a few minutes per 250 samples); the split is then compared with its fingerprint in
# results/datasets/ if there is one. Splits that exist with the same config are skipped.
set -euo pipefail
# shellcheck source=SCRIPTDIR/../lib.sh
source "${HOLOFOCUS_KIT:?submit through cluster/maxwell/submit.sh}/lib.sh"
kit_job_start

config=${1:?usage: datagen CONFIG [--splits SPLIT ...] [--limit N]}
shift
name=$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["name"])' "$config")
out=$HOLOFOCUS_DATA/$name
kit_do holofocus-dataset generate "$config" --out "$out" --device "$HOLOFOCUS_DEVICE" "$@"
status=0
for file in "$out"/*.h5; do
    kit_do holofocus-dataset inspect "$file"
    reference=results/datasets/$name-$(basename "$file" .h5).csv
    if [[ -f $reference ]]; then
        kit_do python scripts/check_cluster_run.py data "$file" --reference "$reference" ||
            status=1
    fi
done
exit "$status"
