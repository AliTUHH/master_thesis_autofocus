#!/bin/bash
# Create the Python environment of the Maxwell run kit (docs/maxwell-gpu.md, section 3):
#
#   cluster/maxwell/setup_env.sh [--with-sbi] [--recreate]
#
# Run it on a login node, since pip needs the internet. It loads HOLOFOCUS_MODULES, creates a
# venv or conda environment in HOLOFOCUS_ENV, installs PyTorch (HOLOFOCUS_TORCH_SPEC from
# HOLOFOCUS_TORCH_INDEX), the package versions of the committed results
# (cluster/maxwell/requirements-vm.txt) and this repository in editable mode, and writes the
# versions to HOLOFOCUS_ENV/holofocus-env.txt. --with-sbi adds the npe extra for Stage 7 (sbi
# at the version of the VM) without changing torch or the pinned packages (pip stops if they
# conflict), --recreate deletes the environment first. With HOLOFOCUS_ENV_KIND=none nothing is
# installed; the script only reports the environment on PATH. pip and conda cache their
# downloads in HOLOFOCUS_SCRATCH/cache unless PIP_CACHE_DIR or CONDA_PKGS_DIRS are set.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=SCRIPTDIR/lib.sh
source "$here/lib.sh"

with_sbi=0 recreate=0
for arg in "$@"; do
    case $arg in
        --with-sbi) with_sbi=1 ;;
        --recreate) recreate=1 ;;
        -h | --help)
            sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p;}' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *) kit_die "unknown option $arg" ;;
    esac
done
kit_load_config
kit_require_checked
kit_load_modules
# By default both caches live in the home directory; the CUDA wheels alone take several GB.
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$HOLOFOCUS_SCRATCH/cache/pip}
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$HOLOFOCUS_SCRATCH/cache/conda-pkgs}

glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null || echo "glibc unknown")
echo "$glibc on $(hostname); the PyTorch wheels need glibc 2.28 or newer"

case $HOLOFOCUS_ENV_KIND in
    venv)
        if ((recreate)); then
            kit_do rm -rf "$HOLOFOCUS_ENV"
        fi
        if [[ ! -d $HOLOFOCUS_ENV ]]; then
            # numpy 2.5.4 and scipy 1.18.1 of requirements-vm.txt need Python 3.12.
            if ! "$HOLOFOCUS_PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 12))'; then
                kit_die "$HOLOFOCUS_PYTHON is older than Python 3.12" \
                    "($("$HOLOFOCUS_PYTHON" -V 2>&1)), which the pinned numpy and scipy need;" \
                    "load a newer Python module or use HOLOFOCUS_ENV_KIND=conda"
            fi
            kit_do "$HOLOFOCUS_PYTHON" -m venv "$HOLOFOCUS_ENV"
        fi
        ;;
    conda)
        if ((recreate)) && [[ -d $HOLOFOCUS_ENV ]]; then
            kit_do "$HOLOFOCUS_CONDA" env remove -y -p "$HOLOFOCUS_ENV"
        fi
        if [[ ! -d $HOLOFOCUS_ENV ]]; then
            kit_do "$HOLOFOCUS_CONDA" create -y -p "$HOLOFOCUS_ENV" python=3.12 pip
        fi
        ;;
esac
kit_activate_env --no-check

if [[ $HOLOFOCUS_ENV_KIND != none ]]; then
    kit_do python -m pip install --upgrade pip
    kit_do python -m pip install "$HOLOFOCUS_TORCH_SPEC" --index-url "$HOLOFOCUS_TORCH_INDEX"
    kit_do python -m pip install -c "$here/requirements-vm.txt" -e "${HOLOFOCUS_REPO}[dev]"
    if ((with_sbi)); then
        pin=$(python -c 'import torch; print(f"torch=={torch.__version__}")')
        kit_do python -m pip install -c "$here/requirements-vm.txt" -c <(echo "$pin") \
            -e "${HOLOFOCUS_REPO}[dev,npe]"
    fi
fi

report=$(
    python -c 'import sys, torch, numpy, holofocus
print("python", sys.version.split()[0], sys.executable)
print("torch", torch.__version__, "CUDA build", torch.version.cuda,
      "cuDNN", torch.backends.cudnn.version())
print("CUDA available here", torch.cuda.is_available(), "(no GPU on a login node is normal)")
print("numpy", numpy.__version__, "holofocus", holofocus.__version__)'
)
echo "$report"
if [[ $HOLOFOCUS_ENV_KIND != none ]]; then
    {
        echo "# $(date '+%Y-%m-%d %H:%M:%S %Z') on $(hostname), $glibc"
        echo "# ${report//$'\n'/$'\n'# }"
        python -m pip freeze
    } >"$HOLOFOCUS_ENV/holofocus-env.txt"
    echo "versions written to $HOLOFOCUS_ENV/holofocus-env.txt"
fi
