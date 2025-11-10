#!/bin/bash
# slurm_scripts/launch_ddp_local.sh
# Usage: launch_ddp_local.sh <your_train_bash_script> [args...]
# Runs the given bash script under torchrun across all local GPUs in the allocation.

GPUS="${GPUS:-4}"           # default: 4 processes for a 4×H100 node
export PYTHONUNBUFFERED=1

# One worker per GPU; run the provided bash script in each worker (no nested sbatch).
torchrun --standalone --no_python --nproc_per_node="${GPUS}" bash "$@"
