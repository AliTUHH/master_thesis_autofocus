# Settings of the Maxwell run kit. Copy this file to cluster/maxwell/config.sh (ignored by
# git) and replace every CHECK_ME; submit.sh and setup_env.sh refuse to run while one is
# left. docs/maxwell-gpu.md, section 2, says how to find each value. "" switches an option
# off. The file is plain bash.
# shellcheck shell=bash disable=SC2034

# Storage. Datasets, the Python environment, run directories and logs go to scratch (large,
# not backed up). Results that must not be lost are copied to DUST (limited) by the jobs.
HOLOFOCUS_SCRATCH="/gpfs/petra3/scratch/CHECK_ME/holofocus"
HOLOFOCUS_DUST="CHECK_ME"

# Python environment, created by setup_env.sh: "venv" (with HOLOFOCUS_PYTHON >= 3.12),
# "conda" (with the conda executable HOLOFOCUS_CONDA) or "none" (use the python on PATH
# after loading HOLOFOCUS_MODULES, for an environment you manage yourself).
HOLOFOCUS_ENV_KIND="venv"
HOLOFOCUS_ENV="$HOLOFOCUS_SCRATCH/env"
HOLOFOCUS_PYTHON="python3"
HOLOFOCUS_CONDA="conda"
# Modules to load before Python is used, separated by spaces; "" for none.
HOLOFOCUS_MODULES="CHECK_ME"

# PyTorch wheels: the CUDA version in the index (cu126, cu130 or cu132 for torch 2.14.1)
# must not be newer than the "CUDA Version" that nvidia-smi shows on the A100 nodes.
HOLOFOCUS_TORCH_INDEX="https://download.pytorch.org/whl/CHECK_ME"
HOLOFOCUS_TORCH_SPEC="torch==2.14.1"

# SLURM. Partition and node feature (--constraint) of the A100 nodes, and the sbatch
# option(s) that request one GPU ("" if a node comes with its GPUs). The CPU partition runs
# jobs submitted with --cpu. HOLOFOCUS_SBATCH_EXTRA is added to every job, e.g. an account.
HOLOFOCUS_GPU_PARTITION="CHECK_ME"
HOLOFOCUS_GPU_CONSTRAINT="CHECK_ME"
HOLOFOCUS_GPU_REQUEST="CHECK_ME"
HOLOFOCUS_CPU_PARTITION="CHECK_ME"
HOLOFOCUS_SBATCH_EXTRA=""
