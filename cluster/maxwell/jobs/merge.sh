#!/bin/bash
#SBATCH --job-name=holofocus-merge
#SBATCH --nodes=1
#SBATCH --time=00:15:00
# Join the chunks of a baseline job array into one run directory:
#
#   submit.sh --cpu --after ARRAY_JOB_ID merge NAME     (or: submit.sh --local merge NAME)
#
# NAME/chunk-*/ are merged into NAME/ (holofocus-baseline merge), which then holds
# results.csv, traces.jsonl, summary.json and run.yaml like a run without chunks. The log
# lists the samples of every chunk; a merge can be repeated after a chunk was completed.
set -euo pipefail
# shellcheck source=SCRIPTDIR/../lib.sh
source "${HOLOFOCUS_KIT:?submit through cluster/maxwell/submit.sh}/lib.sh"
kit_job_start

out=$(kit_run_dir "${1:?usage: merge NAME}")
kit_do holofocus-baseline merge --out "$out" "$out"/chunk-*/
kit_keep "$out"
