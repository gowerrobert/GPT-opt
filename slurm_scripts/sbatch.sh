#!/bin/bash
#SBATCH -p gpu
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --exclude=workergpu094
#SBATCH --output=slurm_logs/slurm_job_%j.out
#SBATCH --error=slurm_logs/slurm_job_%j.err

echo "Starting sbatch.sh script on node $(hostname)..."
echo "Job ID: $SLURM_JOB_ID"
echo "TASK_FILE: $TASK_FILE"

# Ensure the directories exist
echo "Creating log directories if they do not exist..."
mkdir -p slurm_logs disbatch_logs

module load disBatch
echo "Loaded disBatch module."

echo "Activating Conda environment 'ffcv-pl'..."

# Try to source conda setup from standard or user-specified location
if [ -z "$CONDA_SH_PATH" ]; then
    # Default to your known path
    CONDA_SH_PATH="/mnt/home/nghosh/miniforge3/etc/profile.d/conda.sh"
fi

if [ -f "$CONDA_SH_PATH" ]; then
    echo "Sourcing conda setup from $CONDA_SH_PATH"
    source "$CONDA_SH_PATH"
    conda activate ffcv-pl
    echo "Conda environment activated."
else
    echo "ERROR: conda.sh not found at $CONDA_SH_PATH. Please set CONDA_SH_PATH."
    exit 1
fi


echo "Starting disBatch with task file $TASK_FILE and log prefix 'disbatch_logs'..."
disBatch $TASK_FILE --prefix "disbatch_logs"
echo "disBatch job completed for Job ID $SLURM_JOB_ID"

# Wait for all commands to finish
wait

# Move any files starting with slurm_job_${SLURM_JOB_ID} to the specified directory
mv slurm_job_${SLURM_JOB_ID}_* slurm_logs/ 2>/dev/null || true
