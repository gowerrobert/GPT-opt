#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml> [num_gpus]"
  exit 1
fi

CONFIG_FILE="$1"
GPUS="${2:-4}"

if ! [[ "$GPUS" =~ ^[0-9]+$ ]] || [[ "$GPUS" -lt 1 ]]; then
  echo "Error: num_gpus must be a positive integer"
  exit 1
fi

CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
mkdir -p slurm_logs

sbatch <<EOF
#!/bin/bash
#SBATCH -J ${CONFIG_NAME}
#SBATCH --gpus-per-node=${GPUS}
# SBATCH --gpus=${GPUS}
#SBATCH --cpus-per-gpu=8
#SBATCH --time=100:00:00
#SBATCH -C h100
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH -o slurm_logs/${CONFIG_NAME}.log

export OMP_NUM_THREADS=1

# Activate environment
source venv/bin/activate

# Install the necessary packages
python -m pip install -e .

# Run the Python script with the config file
time torchrun --standalone --nproc_per_node=${GPUS} run.py  ${CONFIG_FILE}
EOF