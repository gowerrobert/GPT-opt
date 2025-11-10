#!/bin/bash

SCRIPT=$1
SCRIPT_NAME=$(basename "$SCRIPT")

# Remove the script path from the positional parameters so only user args remain (e.g., lr, wd)
shift
GPUS="${GPUS:-1}"

mkdir -p slurm_logs

sbatch <<EOF
#!/bin/bash
#SBATCH -J ${SCRIPT_NAME}
#SBATCH --gpus=${GPUS}
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=60:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --exclude=workergpu027
#SBATCH -o slurm_logs/%j.log
#SBATCH -e slurm_logs/%j.err

module load python

# Activate environment
source venv/bin/activate

# Install the necessary packages
# python3 -m pip install -e .

export PYTHONUNBUFFERED=1

# Always launch via torchrun with one worker per GPU; run the provided bash script in each worker
torchrun --standalone --no_python --nproc_per_node=${GPUS} bash "${SCRIPT}" $@
EOF
