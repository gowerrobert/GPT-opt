#!/bin/bash
# Note: partition, nodes, ntasks, gpus-per-task, constraint, and job-name are set via sbatch command-line args
#SBATCH --cpus-per-task=16
#SBATCH --time=40:00:00
#SBATCH --output=slurm_logs/slurm_job_%j.out
#SBATCH --error=slurm_logs/slurm_job_%j.err

echo "Starting sbatch_ddp.sh on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "GPUs per task: ${SLURM_GPUS_PER_TASK:-unknown}"
echo "TASK_FILE: ${TASK_FILE:-unset}"

mkdir -p slurm_logs disbatch_logs

module load disBatch || true
module load python   || true
# If you normally activate a venv, keep doing it:
[ -d "venv" ] && source venv/bin/activate

# Each line in $TASK_FILE is a full command that already includes our DDP launcher.
echo "Launching disBatch (1 task == 1 node, DDP local)..."
disBatch "$TASK_FILE" --prefix "disbatch_logs"
echo "disBatch job completed for Job ID $SLURM_JOB_ID"

# Wait for all commands to finish
wait

# Move any files starting with slurm_job_${SLURM_JOB_ID} to the specified directory
mv slurm_job_${SLURM_JOB_ID}_* slurm_logs/ 2>/dev/null || true

