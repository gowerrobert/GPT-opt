#!/bin/bash

# Load project configuration
source project_config.sh

# Check if exactly 4 arguments are provided
if [ "$#" -ne 4 ]; then
    echo "Error: Missing arguments."
    echo "Usage: $0 <bash_script> <param_file> <group_name> <nodes>"
    exit 1
fi

# Define input arguments
bash_script=$1
param_file=$2
group_name=$3
nodes=$4

# Set up environment variables and directories
LOG_DIR="${ROOT_DIR}/logs/${group_name}"
RUN_INFO_DIR="${LOG_DIR}/run_info"

# Display configuration summary
echo "Starting submit.sh script..."
echo "Configuration:"
echo "bash_script: $bash_script"
echo "param_file: $param_file"
echo "group_name: $group_name"
echo "nodes: $nodes"

# Create necessary directories
mkdir -p "$RUN_INFO_DIR"

# Copy essential files to the run_info directory
echo "Copying configuration files..."
cp "$bash_script" "${RUN_INFO_DIR}/train.sh"
cp "$param_file" "${RUN_INFO_DIR}/params.json"

echo "Renaming directories..."
source slurm_scripts/rename_utils.sh

# Call rename_dir and capture the new directory path
new_run_info_dir=$(rename_dir "$RUN_INFO_DIR")

# Check if rename_dir command was successful
if [ $? -ne 0 ]; then
    echo "Renaming directories failed."
    exit 1
fi

# Update RUN_INFO_DIR to the new directory
RUN_INFO_DIR="$new_run_info_dir"
task_file="${RUN_INFO_DIR}/tasks"
run_logs="${RUN_INFO_DIR}/logs"

# Optionally, display the new path
echo "New RUN_INFO_DIR: $RUN_INFO_DIR"

# Generate task file with logging options
echo "Generating task file..."
python generate_task_file.py \
    --bash_script="$bash_script" \
    --param_file="$param_file" \
    --output_file="$task_file" \
    --full_tasks=True \
    --add_logs=True \
    --log_dir="$run_logs"

# Check if task generation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to generate task file. Aborting job submission."
    exit 1
fi

export TASK_FILE=$task_file
export GROUP_NAME=$group_name
export WANDB_RUN_GROUP=$group_name
export LOG_DIR
export RUN_INFO_DIR

# Submit the sbatch job
echo "Submitting sbatch job..."
sbatch --nodes="$nodes" slurm_scripts/sbatch.sh
echo "Job submitted. Check slurm logs for job status."

