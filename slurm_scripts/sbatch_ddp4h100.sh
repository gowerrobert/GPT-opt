#!/bin/bash
#SBATCH -p gpuxl
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --constraint=h100
#SBATCH --output=slurm_logs/slurm_job_%j.out
#SBATCH --error=slurm_logs/slurm_job_%j.err


echo "Starting sbatch_ddp4h100.sh on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "TASK_FILE: ${TASK_FILE:-unset}"

mkdir -p slurm_logs disbatch_logs

module load disBatch || true
module load python   || true
# If you normally activate a venv, keep doing it:
[ -d "venv" ] && source venv/bin/activate

# Each line in $TASK_FILE is a full command that already includes our DDP launcher.
echo "Launching disBatch (1 task == 1 node with 4×H100, DDP local)..."
disBatch "$TASK_FILE" --prefix "disbatch_logs"
