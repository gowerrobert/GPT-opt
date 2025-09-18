#!/bin/bash
# submit_nodes_ddp.sh
#
# Same style as submit.sh but simplified for node-level DDP:
#   Usage: ./submit_nodes_ddp.sh <bash_script> <param_file> <group_name> <count>
#   - <count> is interpreted as number of NODES (each node = 4×H100, DDP)
#
# Respects your existing config/env:
#   - sources project_config.sh
#   - uses RUN_INFO_DIR if you already export/set it elsewhere
#   - otherwise defaults to runs/<group_name> for logs/tasks (no timestamping)
#
# Environment overrides (optional):
#   PARTITION=gpuxl (default)
#   CONSTRAINT=h100 (default)

# Load your project settings like submit.sh does
source project_config.sh

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <bash_script> <param_file> <group_name> <count>"
  echo "Example: $0 run_slim_pajama10B_adam_large.sh params/slim_pajama.json pajama10b_ddp 8"
  exit 1
fi

bash_script="$1"         # e.g., run_slim_pajama10B_adam_large.sh
param_file="$2"          # e.g., params/slim_pajama.json
group_name="$3"          # e.g., pajama10b_ddp
nodes="$4"               # number of nodes to use

partition="${PARTITION:-gpuxl}"
constraint="${CONSTRAINT:-h100}"

# Keep paths simple + compatible with your existing layout:
# If RUN_INFO_DIR is already set by your workflow, honor it.
# Otherwise, use a stable path under runs/<group_name>.
RUN_INFO_DIR="${RUN_INFO_DIR:-runs/${group_name}}"
mkdir -p "${RUN_INFO_DIR}"
mkdir -p slurm_logs disbatch_logs

task_file="${RUN_INFO_DIR}/tasks"
run_logs="${RUN_INFO_DIR}/logs"
mkdir -p "${run_logs}"

echo "=== submit_nodes_ddp.sh ==="
echo "bash_script: ${bash_script}"
echo "param_file:  ${param_file}"
echo "group_name:  ${group_name}"
echo "nodes:       ${nodes}"
echo "partition:   ${partition}"
echo "constraint:  ${constraint}"
echo "RUN_INFO_DIR:${RUN_INFO_DIR}"
echo "task_file:   ${task_file}"

# 1) Generate the task file exactly like your submit.sh flow expects.
python generate_task_file.py \
  --bash_script="${bash_script}" \
  --param_file="${param_file}" \
  --output_file="${task_file}" \
  --full_tasks=True \
  --add_logs=True \
  --log_dir="${run_logs}"

# 2) Prefix each task line to run under torchrun across the 4 local GPUs.
#    (no change to your training scripts themselves)
sed -i 's|^|bash slurm_scripts/launch_ddp_local.sh |' "${task_file}"

# Export the same env your other scripts likely use
export TASK_FILE="${task_file}"
export GROUP_NAME="${group_name}"
export WANDB_RUN_GROUP="${group_name}"

# 3) Submit: one task per node; each task requests 4×H100 and runs DDP locally.
echo "Submitting: nodes=${nodes}, ntasks=${nodes}, partition=${partition}, constraint=${constraint}"
sbatch \
  --partition="${partition}" \
  --constraint="${constraint}" \
  --nodes="${nodes}" \
  --ntasks="${nodes}" \
  --job-name="${group_name}" \
  slurm_scripts/sbatch_ddp4h100.sh
