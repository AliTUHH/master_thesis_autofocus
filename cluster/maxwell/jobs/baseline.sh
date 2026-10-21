#!/bin/bash
#SBATCH --job-name=holofocus-baseline
#SBATCH --nodes=1
#SBATCH --time=24:00:00
# Classical autofocus of [2] on the samples of a split (holofocus-baseline run):
#
#   submit.sh [--array 0-9] baseline NAME CONFIG DATA [--chunks 10] [run options]
#
# NAME is the run directory under $HOLOFOCUS_SCRATCH/runs, CONFIG a file of
# configs/baseline/ and DATA a split file. Options such as --limit N or --warm-start
# PREDICTIONS.csv --half-width 5e-4 are passed on to holofocus-baseline run. In a job array
# task k runs chunk k of the --chunks N contiguous chunks of the selected samples into
# NAME/chunk-k-of-N, so "--array 3 ... --chunks 10" repeats chunk 3 alone; "submit.sh merge
# NAME" joins the chunks afterwards. Runs are resumable: a job that hit its time limit
# continues with the next sample when it is submitted again with the same arguments.
set -euo pipefail
# shellcheck source=SCRIPTDIR/../lib.sh
source "${HOLOFOCUS_KIT:?submit through cluster/maxwell/submit.sh}/lib.sh"
kit_job_start

usage="usage: baseline NAME CONFIG DATA [--chunks N] [holofocus-baseline run options]"
out=$(kit_run_dir "${1:?$usage}")
config=${2:?$usage}
data=$(kit_data "${3:?$usage}")
shift 3
chunks="" previous=""
for arg in "$@"; do
    if [[ $previous == --chunks ]]; then
        chunks=$arg
    elif [[ $arg == --chunks=* ]]; then
        chunks=${arg#--chunks=}
    fi
    previous=$arg
done
if [[ -n ${SLURM_ARRAY_TASK_ID:-} ]]; then
    if [[ -z $chunks ]]; then
        kit_die "a job array needs --chunks N (task k runs chunk k of N)"
    fi
    out=$out/chunk-$SLURM_ARRAY_TASK_ID-of-$chunks
    set -- "$@" --chunk "$SLURM_ARRAY_TASK_ID"
elif [[ -n $chunks ]]; then
    kit_die "--chunks needs a job array (submit.sh --array 0-$((chunks - 1)))"
fi
kit_do holofocus-baseline run "$config" --data "$data" --out "$out" \
    --device "$HOLOFOCUS_DEVICE" --threads "$OMP_NUM_THREADS" "$@"
kit_keep "$out"
