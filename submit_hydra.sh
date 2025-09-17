#!/bin/bash

SCRIPT=$1
SCRIPT_NAME=$(basename "$SCRIPT")

# Remove the script path from the positional parameters so only user args remain (e.g., lr, wd)
shift

mkdir -p output/slurm_logs

sbatch <<EOF
#!/bin/bash
#SBATCH -J ${SCRIPT_NAME}
#SBATCH --gpus=1
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

# Run the Python script with the config file
srun -u bash "$SCRIPT" $@
EOF
