#!/bin/bash
# submit_nodes_ddp.sh
#
# Same style as submit.sh but simplified for node-level DDP:
#   Usage: ./submit_nodes_ddp.sh <bash_script> <param_file> <group_name> <count> [--num_gpus=N] [--partition=P] [--constraint=C]
#   - <count> is interpreted as number of NODES
#   - --num_gpus is optional, defaults to 4 (can be 4 or 8)
#   - --partition is optional, defaults to gpuxl (can be gpuxl or gpu)
#   - --constraint is optional (e.g., h100, a100)
#
# Respects your existing config/env:
#   - sources project_config.sh
#   - uses RUN_INFO_DIR if you already export/set it elsewhere
#   - otherwise defaults to runs/<group_name> for logs/tasks (no timestamping)
#

# Load your project settings like submit.sh does
source project_config.sh

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <bash_script> <param_file> <group_name> <count> [--num_gpus=N] [--partition=P] [--constraint=C]"
  echo "Example: $0 scripts/pretrain_gpt.sh params/params.json my_exp 8"
  echo "Example: $0 scripts/pretrain_gpt.sh params/params.json my_exp 4 --num_gpus=8 --partition=gpu"
  exit 1
fi

# Parse required positional arguments
bash_script="$1"         # e.g., run_slim_pajama10B_adam_large.sh
param_file="$2"          # e.g., params/slim_pajama.json
group_name="$3"          # e.g., pajama10b_ddp
nodes="$4"               # number of nodes to use

# Set defaults for optional arguments
num_gpus=4
partition="${PARTITION:-gpuxl}"
constraint=""

# Parse optional keyword arguments
shift 4
while [[ $# -gt 0 ]]; do
  case $1 in
    --num_gpus=*)
      num_gpus="${1#*=}"
      shift
      ;;
    --partition=*)
      partition="${1#*=}"
      shift
      ;;
    --constraint=*)
      constraint="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 <bash_script> <param_file> <group_name> <count> [--num_gpus=N] [--partition=P] [--constraint=C]"
      exit 1
      ;;
  esac
done

# Align with submit.sh: use ROOT_DIR/logs/<group_name>/run_info and copy inputs
LOG_DIR="${ROOT_DIR}/logs/${group_name}"
RUN_INFO_DIR="${LOG_DIR}/run_info"
mkdir -p "${RUN_INFO_DIR}"

echo "Copying configuration files..."
cp "${bash_script}" "${RUN_INFO_DIR}/train.sh"
cp "${param_file}" "${RUN_INFO_DIR}/params.json"

echo "Renaming directories..."
source slurm_scripts/rename_utils.sh
new_run_info_dir=$(rename_dir "${RUN_INFO_DIR}")
if [ $? -ne 0 ]; then
  echo "Renaming directories failed."
  exit 1
fi

RUN_INFO_DIR="${new_run_info_dir}"
task_file="${RUN_INFO_DIR}/tasks"
run_logs="${RUN_INFO_DIR}/logs"

echo "=== submit_nodes_ddp.sh ==="
echo "bash_script: ${bash_script}"
echo "param_file:  ${param_file}"
echo "group_name:  ${group_name}"
echo "nodes:       ${nodes}"
echo "num_gpus:    ${num_gpus}"
echo "partition:   ${partition}"
echo "constraint:  ${constraint:-none}"
echo "LOG_DIR:     ${LOG_DIR}"
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

if [ $? -ne 0 ]; then
  echo "Error: Failed to generate task file. Aborting job submission."
  exit 1
fi

# 2) Prefix each task line to run under torchrun across the local GPUs.
#    (no change to your training scripts themselves)
sed -i "s|^|GPUS=${num_gpus} bash slurm_scripts/launch_ddp_local.sh |" "${task_file}"

# Export the same env your other scripts likely use
export TASK_FILE="${task_file}"
export GROUP_NAME="${group_name}"
export WANDB_RUN_GROUP="${group_name}"
export LOG_DIR
export RUN_INFO_DIR
export NUM_GPUS="${num_gpus}"
export CONSTRAINT="${constraint}"

# 3) Submit: one task per node; each task requests N GPUs and runs DDP locally.
echo "Submitting: partition=${partition}, nodes=${nodes}, ntasks=${nodes}, gpus_per_task=${num_gpus}"
sbatch_args=(
  --partition="${partition}"
  --nodes="${nodes}"
  --ntasks="${nodes}"
  --gpus-per-task="${num_gpus}"
  --job-name="${group_name}"
)

# Add constraint if specified
if [ -n "${constraint}" ]; then
  sbatch_args+=(--constraint="${constraint}")
fi

sbatch "${sbatch_args[@]}" slurm_scripts/sbatch_ddp.sh
