# Shared settings and functions of the Maxwell run kit (docs/maxwell-gpu.md). Sourced by
# submit.sh, setup_env.sh and the job scripts, not run directly.
# shellcheck shell=bash

HOLOFOCUS_KIT=${HOLOFOCUS_KIT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
HOLOFOCUS_REPO=$(cd "$HOLOFOCUS_KIT/../.." && pwd)
KIT_PLACEHOLDER=CHECK_ME
KIT_SETTINGS=(
    HOLOFOCUS_SCRATCH HOLOFOCUS_DUST
    HOLOFOCUS_ENV_KIND HOLOFOCUS_ENV HOLOFOCUS_PYTHON HOLOFOCUS_CONDA HOLOFOCUS_MODULES
    HOLOFOCUS_TORCH_INDEX HOLOFOCUS_TORCH_SPEC
    HOLOFOCUS_GPU_PARTITION HOLOFOCUS_GPU_CONSTRAINT HOLOFOCUS_GPU_REQUEST
    HOLOFOCUS_CPU_PARTITION HOLOFOCUS_SBATCH_EXTRA
)

kit_die() {
    echo "error: $*" >&2
    exit 1
}

# Read config.sh (or $HOLOFOCUS_CONFIG) and derive the scratch layout.
kit_load_config() {
    HOLOFOCUS_CONFIG=${HOLOFOCUS_CONFIG:-$HOLOFOCUS_KIT/config.sh}
    if [[ ! -r $HOLOFOCUS_CONFIG ]]; then
        kit_die "no settings file $HOLOFOCUS_CONFIG; copy cluster/maxwell/config.example.sh" \
            "to cluster/maxwell/config.sh and fill it in"
    fi
    # shellcheck source=SCRIPTDIR/config.example.sh
    source "$HOLOFOCUS_CONFIG"
    local name
    for name in "${KIT_SETTINGS[@]}"; do
        if [[ -z ${!name+set} ]]; then
            kit_die "$HOLOFOCUS_CONFIG does not set $name (see config.example.sh)"
        fi
    done
    HOLOFOCUS_SCRATCH=${HOLOFOCUS_SCRATCH%/}
    HOLOFOCUS_DATA=$HOLOFOCUS_SCRATCH/data
    HOLOFOCUS_RUNS=$HOLOFOCUS_SCRATCH/runs
    HOLOFOCUS_LOGS=$HOLOFOCUS_SCRATCH/logs
    export HOLOFOCUS_KIT HOLOFOCUS_REPO HOLOFOCUS_CONFIG
    export HOLOFOCUS_DATA HOLOFOCUS_RUNS HOLOFOCUS_LOGS
}

# Names of the settings that still hold the placeholder.
kit_unchecked() {
    local name
    for name in "${KIT_SETTINGS[@]}"; do
        if [[ ${!name} == *"$KIT_PLACEHOLDER"* ]]; then
            echo "$name"
        fi
    done
}

kit_require_checked() {
    local left
    left=$(kit_unchecked)
    if [[ -n $left ]]; then
        kit_die "replace $KIT_PLACEHOLDER in $HOLOFOCUS_CONFIG first (docs/maxwell-gpu.md," \
            "section 2):" "${left//$'\n'/ }"
    fi
}

# Module init scripts often reference unset variables, hence set +u around them.
kit_load_modules() {
    if [[ -z $HOLOFOCUS_MODULES ]]; then
        return 0
    fi
    set +u
    if ! type module >/dev/null 2>&1; then
        local init
        for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh \
            /usr/share/lmod/lmod/init/bash /usr/share/Modules/init/bash; do
            if [[ -r $init ]]; then
                # shellcheck disable=SC1090
                source "$init"
                break
            fi
        done
    fi
    if ! type module >/dev/null 2>&1; then
        kit_die "HOLOFOCUS_MODULES is set, but there is no module command"
    fi
    # shellcheck disable=SC2086
    module load $HOLOFOCUS_MODULES
    set -u
}

# kit_activate_env [--no-check]: activate HOLOFOCUS_ENV; the check for holofocus is skipped
# before it is installed.
kit_activate_env() {
    set +u
    case $HOLOFOCUS_ENV_KIND in
        venv)
            if [[ ! -r $HOLOFOCUS_ENV/bin/activate ]]; then
                kit_die "no venv in $HOLOFOCUS_ENV; run cluster/maxwell/setup_env.sh"
            fi
            # shellcheck disable=SC1091
            source "$HOLOFOCUS_ENV/bin/activate"
            ;;
        conda)
            eval "$("$HOLOFOCUS_CONDA" shell.bash hook)"
            conda activate "$HOLOFOCUS_ENV"
            ;;
        none) ;;
        *) kit_die "HOLOFOCUS_ENV_KIND must be venv, conda or none, not '$HOLOFOCUS_ENV_KIND'" ;;
    esac
    set -u
    if [[ ${1:-} != --no-check ]] && ! command -v holofocus-baseline >/dev/null; then
        kit_die "holofocus is not installed in this environment; run cluster/maxwell/setup_env.sh"
    fi
}

# Start of every job: settings, modules, environment, threads, device and a header.
kit_job_start() {
    kit_load_config
    kit_load_modules
    kit_activate_env
    cd "$HOLOFOCUS_REPO" || kit_die "cannot enter $HOLOFOCUS_REPO"
    HOLOFOCUS_DEVICE=${HOLOFOCUS_DEVICE:-cuda}
    unset OMP_NUM_THREADS
    OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-$(nproc)}
    # TF32 would change float32 convolutions on the A100 at the 1e-3 level; the checks
    # against the CPU results need full float32.
    NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE:-0}
    export HOLOFOCUS_DEVICE OMP_NUM_THREADS NVIDIA_TF32_OVERRIDE
    kit_header
    if [[ $HOLOFOCUS_DEVICE == cuda* ]] &&
        ! python -c 'import sys, torch; sys.exit(not torch.cuda.is_available())'; then
        kit_die "CUDA is not available on $(hostname); see docs/maxwell-gpu.md, section 10"
    fi
}

kit_header() {
    local commit
    commit=$(git -C "$HOLOFOCUS_REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
    if ! git -C "$HOLOFOCUS_REPO" diff --quiet HEAD 2>/dev/null; then
        commit="$commit (with local changes)"
    fi
    echo "== holofocus job ${SLURM_JOB_NAME:-local} ${SLURM_JOB_ID:-}" \
        "${SLURM_ARRAY_TASK_ID:+array task $SLURM_ARRAY_TASK_ID}"
    echo "date      $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "host      $(hostname), partition ${SLURM_JOB_PARTITION:-none}, $(nproc) cpus," \
        "OMP_NUM_THREADS=$OMP_NUM_THREADS"
    echo "repo      $HOLOFOCUS_REPO at $commit"
    echo "settings  $HOLOFOCUS_CONFIG"
    echo "device    $HOLOFOCUS_DEVICE"
    if command -v nvidia-smi >/dev/null; then
        nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader |
            sed 's/^/gpu       /' || true
    fi
    python -c 'import sys, torch, holofocus
print("python   ", sys.version.split()[0], "| torch", torch.__version__, "| CUDA build",
      torch.version.cuda, "| CUDA available", torch.cuda.is_available(), "| holofocus",
      holofocus.__version__)'
}

# Print a command, then run it.
kit_do() {
    echo "+ $(printf '%q ' "$@")"
    "$@"
}

# Paths relative to the scratch layout: data files and run directories.
kit_data() {
    if [[ $1 == /* ]]; then echo "$1"; else echo "$HOLOFOCUS_DATA/$1"; fi
}

kit_run_dir() {
    if [[ $1 == /* ]]; then echo "$1"; else echo "$HOLOFOCUS_RUNS/$1"; fi
}

# kit_keep DIR: copy a run directory without HDF5 files to DUST, at the same place relative
# to $HOLOFOCUS_DUST/holofocus as DIR is relative to $HOLOFOCUS_SCRATCH.
kit_keep() {
    local src dest
    src=$(cd "$1" && pwd)
    if [[ -z $HOLOFOCUS_DUST ]]; then
        echo "keep: HOLOFOCUS_DUST is empty, $src stays on scratch only"
        return 0
    fi
    if [[ $src == "$HOLOFOCUS_SCRATCH"/* ]]; then
        dest=$HOLOFOCUS_DUST/holofocus/${src#"$HOLOFOCUS_SCRATCH"/}
    else
        dest=$HOLOFOCUS_DUST/holofocus/$(basename "$src")
    fi
    if ! (mkdir -p "$dest" &&
        tar -C "$src" --exclude='*.h5' --exclude='*.part' -cf - . | tar -C "$dest" -xf -); then
        kit_die "keep: copying $src to $dest failed (DUST quota? df -h $HOLOFOCUS_DUST);" \
            "the results are still in $src"
    fi
    echo "keep: $src -> $dest ($(du -sh "$dest" | cut -f1))"
}
