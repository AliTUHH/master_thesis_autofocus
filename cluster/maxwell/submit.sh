#!/bin/bash
# Submit a job of the Maxwell run kit to SLURM (docs/maxwell-gpu.md).
#
#   cluster/maxwell/submit.sh [options] JOB [job arguments]
#
# JOB is a script in cluster/maxwell/jobs/ (gpu-check, datagen, predict, baseline, merge,
# train, run); its head lists the arguments. Config files are relative to the repository,
# data files relative to $HOLOFOCUS_SCRATCH/data and run directories relative to
# $HOLOFOCUS_SCRATCH/runs. Jobs run with --device cuda on the A100 nodes, or with
# --device cpu on the CPU partition (--cpu). Logs go to $HOLOFOCUS_SCRATCH/logs.
#
# Options:
#   --cpu          CPU partition, no GPU request, --device cpu
#   --time T       time limit, e.g. 04:00:00 (default: #SBATCH --time of the job)
#   --array SPEC   job array, e.g. 0-9 or 3 (baseline: task k runs chunk k of --chunks N;
#                  train: task k trains with --seed k)
#   --after ID     start when job ID (all tasks of an array) has finished successfully
#   --sbatch OPT   extra sbatch option, e.g. --sbatch --mem=64G (repeatable)
#   --dry-run      print the sbatch command instead of submitting it
#   --local        run the job in this shell without SLURM (tests; --device cpu unless
#                  HOLOFOCUS_DEVICE is set)
#   -h, --help     this text
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=SCRIPTDIR/lib.sh
source "$here/lib.sh"

usage() {
    sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p;}' "${BASH_SOURCE[0]}"
}

cpu=0 dry_run=0 local_run=0 array=0
options=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu) cpu=1 ;;
        --time)
            options+=("--time=${2:?--time needs a value}")
            shift
            ;;
        --array)
            options+=("--array=${2:?--array needs a value}")
            array=1
            shift
            ;;
        --after)
            options+=("--dependency=afterok:${2:?--after needs a job id}")
            shift
            ;;
        --sbatch)
            options+=("${2:?--sbatch needs an option}")
            shift
            ;;
        --dry-run) dry_run=1 ;;
        --local) local_run=1 ;;
        -h | --help)
            usage
            exit 0
            ;;
        -*) kit_die "unknown option $1 (see --help)" ;;
        *) break ;;
    esac
    shift
done
if [[ $# -lt 1 ]]; then
    usage >&2
    exit 2
fi
job=$1
shift
script=$here/jobs/$job.sh
if [[ ! -f $script ]]; then
    known=()
    for path in "$here"/jobs/*.sh; do
        known+=("$(basename "$path" .sh)")
    done
    kit_die "unknown job '$job'; jobs:" "${known[*]}"
fi
kit_load_config
kit_require_checked

if ((local_run)); then
    HOLOFOCUS_DEVICE=${HOLOFOCUS_DEVICE:-cpu} exec bash "$script" "$@"
fi

if ((array)); then
    log=$HOLOFOCUS_LOGS/%x-%A_%a.out
else
    log=$HOLOFOCUS_LOGS/%x-%j.out
fi
command=(sbatch --output="$log" --chdir="$HOLOFOCUS_REPO")
if ((cpu)); then
    device=cpu
    command+=(--partition="$HOLOFOCUS_CPU_PARTITION")
else
    device=cuda
    command+=(--partition="$HOLOFOCUS_GPU_PARTITION")
    if [[ -n $HOLOFOCUS_GPU_CONSTRAINT ]]; then
        command+=(--constraint="$HOLOFOCUS_GPU_CONSTRAINT")
    fi
    # shellcheck disable=SC2206
    command+=($HOLOFOCUS_GPU_REQUEST)
fi
# shellcheck disable=SC2206
command+=($HOLOFOCUS_SBATCH_EXTRA ${options[@]+"${options[@]}"})
command+=("--export=ALL,HOLOFOCUS_KIT=$HOLOFOCUS_KIT,HOLOFOCUS_CONFIG=$HOLOFOCUS_CONFIG,HOLOFOCUS_DEVICE=$device")
command+=("$script" "$@")
if ((dry_run)); then
    printf '%q ' "${command[@]}"
    echo
    exit 0
fi
mkdir -p "$HOLOFOCUS_LOGS"
"${command[@]}"
