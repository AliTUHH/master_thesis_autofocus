#!/bin/bash
#SBATCH --job-name=holofocus-npe
#SBATCH --nodes=1
#SBATCH --time=12:00:00
# Train a neural posterior estimator of dz01 (Stage 7), then predict its validation split and
# optionally a test split and a calibration split with sbi's checks on it:
#
#   submit.sh [--array 0-K] npe NAME CONFIG --train DATA ... --val DATA [--test DATA]
#                              [--calibration DATA] [holofocus-learn npe-train options]
#
# NAME is the run directory under $HOLOFOCUS_SCRATCH/runs, CONFIG an NPE file of
# configs/learned/ (npe-nsf.yaml, npe-mdn.yaml) and DATA split files. Options such as
# --limit N or --seed S are passed on. In a job array task k trains with --seed k into
# NAME-seedk (an ensemble). The environment needs sbi (setup_env.sh --with-sbi). The run
# directory is copied to DUST at the end.
set -euo pipefail
# shellcheck source=SCRIPTDIR/../lib.sh
source "${HOLOFOCUS_KIT:?submit through cluster/maxwell/submit.sh}/lib.sh"
kit_job_start

usage="usage: npe NAME CONFIG --train DATA ... --val DATA [--test DATA] [--calibration DATA]"
out=$(kit_run_dir "${1:?$usage}")
config=${2:?$usage}
shift 2
train=() val="" test="" calibration="" rest=()
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
        --calibration)
            calibration=$(kit_data "${2:?$usage}")
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
device=(--device "$HOLOFOCUS_DEVICE")
kit_do holofocus-learn npe-train "$config" --train "${train[@]}" --val "$val" --out "$out" \
    "${device[@]}" ${rest[@]+"${rest[@]}"}
kit_do holofocus-learn npe-predict "$out" --data "$val" --out "$out/val" "${device[@]}"
if [[ -n $test ]]; then
    kit_do holofocus-learn npe-predict "$out" --data "$test" --out "$out/test" "${device[@]}"
fi
if [[ -n $calibration ]]; then
    kit_do holofocus-learn npe-predict "$out" --data "$calibration" --out "$out/calibration" \
        "${device[@]}"
    kit_do holofocus-learn npe-check "$out" --data "$calibration" --out "$out/calibration" \
        "${device[@]}"
fi
kit_keep "$out"
