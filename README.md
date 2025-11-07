# GPT-opt

Testing optimization methods for training GPT models using Hydra configuration and SLURM scheduling.

---

## Setup

```bash
./setup_env.sh
wandb login  # optional, for experiment tracking
```

---

## Running Experiments

### Local Runs

```bash
# Basic run with Hydra
python run_hydra.py model=gpt-small optimizer=adamw data=shakespeare

# Override specific parameters
python run_hydra.py \
    model=gpt-medium \
    optimizer=adamw \
    optimizer.optimizer_params.lr=0.001 \
    training.training_params.batch_size=32
```

**Outputs:** `outputs/<model>/<dataset>/<optimizer>/<run_name>/`

### SLURM Submissions

#### 1. Standard Parameter Sweeps (`submit.sh`)

For running grid searches across multiple GPUs:

```bash
./slurm_scripts/submit.sh \
    scripts/run_slim_pajama10B_adam_medium.sh \
    param_configs/adamw.json \
    experiment_name \
    8  # number of GPUs
```

**Example with partition:**
```bash
PARTITION=gpu ./slurm_scripts/submit.sh scripts/run_*.sh params.json exp_name 4
```

#### 2. Multi-Node DDP Training (`submit_nodes_ddp.sh`)

For distributed training across multiple nodes:

```bash
./slurm_scripts/submit_nodes_ddp.sh \
    scripts/run_slim_pajama10B_adam_large.sh \
    param_configs/adamw.json \
    experiment_name \
    4  # number of nodes
```

**With custom GPUs/partition/constraint:**
```bash
./slurm_scripts/submit_nodes_ddp.sh \
    scripts/run_*.sh \
    params.json \
    exp_name \
    8 \
    --num_gpus=8 \
    --partition=gpu \
    --constraint=a100
```

**Options:**
- `--num_gpus=N` - GPUs per node (default: 4)
- `--partition=P` - SLURM partition (default: gpuxl)
- `--constraint=C` - GPU constraint (e.g., h100, a100)

---

## Configuration System

### Hydra Config Structure

```
hydra_conf/
├── model/          # gpt-small, gpt-medium, gpt-large
├── optimizer/      # adamw, dap-ns-nadam, etc.
├── training/       # slim_pajama10B, fineweb1B, etc.
├── data/           # Dataset configurations
└── logging/        # default, wandb
```

### Selecting Configs

```bash
python run_hydra.py \
    model=gpt-large \           # Use model/gpt-large.yaml
    optimizer=adamw \            # Use optimizer/adamw.yaml
    training=slim_pajama10B \    # Use training/slim_pajama10B.yaml
    data=slim_pajama10B          # Use data/slim_pajama10B.yaml
```

### Example Config Files

**`hydra_conf/model/gpt-small.yaml`**
```yaml
config:
  n_embd: 768
  n_layer: 12
  n_head: 12
  flash_attention: True
name: gpt-small
```

**`hydra_conf/optimizer/adamw.yaml`**
```yaml
optimizer_params:
  name: adamw
  lr: 0.001
  weight_decay: 0
  lr_schedule: constant-linear
```

---

## Parameter Sweeps

### 1. Create Parameter File

**`param_configs/my_sweep.json`**
```json
{
    "lr": [0.0001, 0.0003, 0.001],
    "wd": [0.0, 0.01, 0.1]
}
```

### 2. Create Training Script

**`scripts/my_experiment.sh`**
```bash
#!/bin/bash
lr=${1:-0.0001}
wd=${2:-0.1}

python run_hydra.py \
    model=gpt-medium \
    optimizer=adamw \
    optimizer.optimizer_params.lr=$lr \
    optimizer.optimizer_params.weight_decay=$wd
```

### 3. Submit Sweep

```bash
./slurm_scripts/submit.sh \
    scripts/my_experiment.sh \
    param_configs/my_sweep.json \
    my_sweep_name \
    9  # Run all 9 combinations (3×3) in parallel
```

This generates all parameter combinations and runs them in parallel. Each run is logged separately.

---

## Directory Structure

### Key Directories

```
GPT-opt/
├── scripts/              # Training wrapper scripts
├── slurm_scripts/        # SLURM submission scripts
├── param_configs/        # Parameter sweep JSON files
├── hydra_conf/          # Modular Hydra configs
├── gptopt/              # Python package (models, optimizers, training)
├── logs/                # SLURM job metadata & task outputs
├── slurm_logs/          # SLURM system logs (.out, .err)
├── outputs/             # Training results & checkpoints
└── disbatch_logs/       # Task distribution logs
```

### Log Organization

**When you submit a job, logs are organized as:**

```
logs/<experiment_name>/
└── run_info_N/              # Auto-incremented for each submission
    ├── train.sh             # Copy of your training script
    ├── params.json          # Copy of parameter file
    ├── tasks                # Generated task file
    └── logs/                # Per-task outputs
        ├── log_0.out
        ├── log_0.err
        └── ...
```

**Training outputs go to:**
```
outputs/<model>/<dataset>/<optimizer>/<run_name>/
├── .hydra/config.yaml       # Full config snapshot
├── task.log                 # Training log
├── wandb/                   # WandB files (if enabled)
└── <optimizer>-*.json       # Training metrics
```

**SLURM system logs:**
```
slurm_logs/
├── slurm_job_<id>.out
└── slurm_job_<id>.err
```

---

## WandB Integration

### Enable WandB Logging

```bash
python run_hydra.py \
    model=gpt-small \
    logging=wandb  # Use logging/wandb.yaml
```

### Configure WandB

Edit `hydra_conf/logging/wandb.yaml`:
```yaml
logging_params:
  wandb:
    project: "my_project"
    name: ${model.name}_${optimizer.optimizer_params.name}
```

### Grouped Runs

When using `submit.sh` or `submit_nodes_ddp.sh`, all parameter combinations are automatically grouped in WandB under the experiment name for easy comparison.

---

## Useful Commands

### Monitor Jobs

```bash
# Check job status
squeue --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R" --me

# Watch live
watch -n 2 'squeue --me'

# View task output
tail -f logs/<experiment_name>/run_info_N/logs/log_0.out
```

### Cancel Jobs

```bash
scancel <job_id>                    # Cancel specific job
scancel --name=<experiment_name>    # Cancel by name
scancel -u $USER                    # Cancel all your jobs
```

### Interactive SLURM Session

```bash
srun --gpus=1 --cpus-per-gpu=8 --time=4:00:00 --partition=gpu --pty bash
module load python
source venv/bin/activate
```

---

## Script Organization

### `scripts/` - Training Wrappers

Bash scripts that call `run_hydra.py` with predefined configurations. Accept parameters as arguments.

**Example:** `scripts/run_slim_pajama10B_adam_medium.sh`
```bash
#!/bin/bash
lr=${1:-0.0001}
wd=${2:-0.1}

python run_hydra.py \
    model=gpt-medium \
    optimizer=adamw \
    training=slim_pajama10B \
    data=slim_pajama10B \
    optimizer.optimizer_params.lr=$lr \
    optimizer.optimizer_params.weight_decay=$wd
```

### `slurm_scripts/` - SLURM Infrastructure

| Script | Purpose |
|--------|---------|
| `submit.sh` | Submit standard parameter sweeps |
| `submit_nodes_ddp.sh` | Submit multi-node DDP training |
| `sbatch.sh` | Execute standard GPU jobs |
| `sbatch_ddp.sh` | Execute multi-node DDP jobs |
| `launch_ddp_local.sh` | Launch torchrun for local DDP |
| `rename_utils.sh` | Auto-increment run_info directories |

---

## How It Works

### Standard Workflow (`submit.sh`)

1. Creates `logs/<experiment_name>/run_info/` directory
2. Copies training script and param file
3. Renames to `run_info_N` (auto-incrementing)
4. Generates task file with all parameter combinations
5. Submits to SLURM via `sbatch.sh`
6. Uses `disBatch` to distribute tasks across GPUs
7. Each task's output saved to `logs/<experiment_name>/run_info_N/logs/`

### Multi-Node DDP Workflow (`submit_nodes_ddp.sh`)

1. Same setup as standard workflow
2. Prefixes each task with `launch_ddp_local.sh`
3. Submits to SLURM via `sbatch_ddp.sh`
4. Each node runs one task using torchrun with N GPUs
5. PyTorch DDP handles distributed training coordination

---

## Quick Reference

| Task | Command |
|------|---------|
| Run locally | `python run_hydra.py model=gpt-small optimizer=adamw data=shakespeare` |
| Submit sweep | `./slurm_scripts/submit.sh scripts/run_*.sh params.json exp_name 8` |
| Submit DDP | `./slurm_scripts/submit_nodes_ddp.sh scripts/run_*.sh params.json exp_name 4` |
| Custom GPUs/partition | Add `--num_gpus=8 --partition=gpu --constraint=h100` |
| Override param | `python run_hydra.py optimizer.optimizer_params.lr=0.001` |
| Check jobs | `squeue --me` |
| View logs | `tail -f logs/<exp>/run_info_N/logs/log_0.out` |

---

## Legacy (Pre-Hydra)

```bash
# Old YAML-based config
python run.py --config configs/shakespeare.yaml
```

---

**For questions or issues, check the code or contact the maintainers.**
