#!/bin/bash
#SBATCH --job-name=holofocus-run
#SBATCH --nodes=1
#SBATCH --time=08:00:00
# Any command in the environment of the kit, e.g. the NPE runs of a later stage:
#
#   submit.sh [--cpu] run COMMAND [ARGUMENTS]
#
# The command runs in the repository with HOLOFOCUS_DATA, HOLOFOCUS_RUNS and
# HOLOFOCUS_DEVICE set; to use them in the arguments, wrap the command in
# bash -c '...' (single quotes, so that they expand in the job). Nothing is copied to DUST.
set -euo pipefail
# shellcheck source=SCRIPTDIR/../lib.sh
source "${HOLOFOCUS_KIT:?submit through cluster/maxwell/submit.sh}/lib.sh"
kit_job_start

if [[ $# -eq 0 ]]; then
    kit_die "usage: run COMMAND [ARGUMENTS]"
fi
kit_do "$@"
