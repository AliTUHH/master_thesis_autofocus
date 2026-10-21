#!/bin/bash
#SBATCH --job-name=holofocus-train
#SBATCH --nodes=1
#SBATCH --time=12:00:00
# Train a learned model, then predict its validation split and optionally a test split:
#
#   submit.sh [--array 0-K] train NAME CONFIG --train DATA ... --val DATA [--test DATA]
#                                [holofocus-learn train options]
#
# NAME is the run directory under $HOLOFOCUS_SCRATCH/runs, CONFIG a file of configs/learned/
# and DATA split files. Options such as --limit N or --seed S are passed on. In a job array
# task k trains with --seed k into NAME-seedk (an ensemble). The run directory is copied to
# DUST at the end.
set -euo pipefail
# shellcheck source=SCRIPTDIR/../lib.sh
source "${HOLOFOCUS_KIT:?submit through cluster/maxwell/submit.sh}/lib.sh"
kit_job_start

usage="usage: train NAME CONFIG --train DATA ... --val DATA [--test DATA] [options]"
out=$(kit_run_dir "${1:?$usage}")
config=${2:?$usage}
shift 2
train=() val="" test="" rest=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --train)
            shift
            while [[ $# -gt 0 && $1 != --* ]]; do
                train+=("$(kit_data "$1")")
                shift
            done
            continue
            ;;
        --val)
            val=$(kit_data "${2:?$usage}")
            shift
            ;;
        --test)
            test=$(kit_data "${2:?$usage}")
            shift
            ;;
        *) rest+=("$1") ;;
    esac
    shift
done
if [[ ${#train[@]} -eq 0 || -z $val ]]; then
    kit_die "$usage"
fi
if [[ -n ${SLURM_ARRAY_TASK_ID:-} ]]; then
    out=$out-seed$SLURM_ARRAY_TASK_ID
    rest+=(--seed "$SLURM_ARRAY_TASK_ID")
fi
kit_do holofocus-learn train "$config" --train "${train[@]}" --val "$val" --out "$out" \
    --device "$HOLOFOCUS_DEVICE" ${rest[@]+"${rest[@]}"}
kit_do holofocus-learn predict "$out" --data "$val" --out "$out/val" --device "$HOLOFOCUS_DEVICE"
if [[ -n $test ]]; then
    kit_do holofocus-learn predict "$out" --data "$test" --out "$out/test" \
        --device "$HOLOFOCUS_DEVICE"
fi
kit_keep "$out"
