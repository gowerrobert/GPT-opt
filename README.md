# GPT-opt

Testing optimization methods for training GPT models using Hydra configuration and SLURM scheduling.

---

## Setup

```bash
./setup_env.sh
wandb login  # optional, for experiment tracking
./tests/test_hydra.sh  #run a small example on gpt-tiny with shakespeare data
```

---

## Running Experiments
### Local Runs

```bash
# Basic run with Hydra
python run_hydra.py model=gpt-tiny optimizer=adamw data=shakespeare training=shakespeare

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

Hydra lets us assemble experiment configurations from small, composable pieces and override them at the command line without editing YAML files. Each config group in `hydra_conf/` corresponds to a different aspect of the training stack (model architecture, optimizer, dataset, logging, etc.). Hydra merges the selected configs into a single runtime configuration, snapshots it in `outputs/<run>/.hydra/`, and makes every field accessible to Python code via the OmegaConf API.

For a deeper dive into how Hydra works, common patterns, and the reasoning behind our layout, see the dedicated guide in `docs/hydra.md`.

### Hydra Config Structure

```
hydra_conf/
├── model/          # gpt-small, gpt-medium, gpt-large
├── optimizer/      # adamw, dap-ns-nadam, etc.
├── training/       # slim_pajama10B, fineweb1B, etc.
├── data/           # Dataset configurations
└── logging/        # default, wandb
```

Each directory is a config group. The default choice for a group is defined in `hydra_conf/config.yaml`, but CLI overrides let you swap any component on the fly. You can also enable additional configs without replacing the defaults by prefixing with `+`, which is useful for stacking logging or callback configs.

### Selecting Configs

```bash
python run_hydra.py \
    model=gpt-large \           # Use model/gpt-large.yaml
    optimizer=adamw \            # Use optimizer/adamw.yaml
    training=slim_pajama10B \    # Use training/slim_pajama10B.yaml
    data=slim_pajama10B          # Use data/slim_pajama10B.yaml
```

Hydra parses the dotted overrides (`optimizer.optimizer_params.lr=...`) and updates only that field, leaving the rest of the config untouched. You can chain as many overrides as needed, and they are type-checked against the schema in the YAML files.

### Multirun Sweeps

Hydra's multirun mode (`-m`) generates cartesian products of parameter values—handy for local sweeps before scaling out to SLURM:

```bash
python run_hydra.py -m \
    optimizer.optimizer_params.lr=0.0003,0.001,0.003 \
    training.training_params.batch_size=16,32
```

Runs are numbered under `multirun/<timestamp>/` and each one captures the exact config it used. The SLURM sweep scripts reuse the same idea under the hood by materializing all override combinations.

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


## Examples GPT-tiny
`
./slurm_scripts/submit.sh  scripts/run_shakespeare_rehpdhg_restart_adamw.sh param_configs/attn_rehpdhg_restart_adamw_sweep.json gpt_tiny_rehpdhg 16
`

`
./slurm_scripts/submit.sh  scripts/run_shakespeare_fista_adamw.sh param_configs/attn_pd_adamw.json gpt_tiny_fista  16
`

`
./slurm_scripts/submit.sh scripts/run_shakespeare_adam.sh param_configs/adamw.json gpt_tiny_adamw 5
`

## Examples GPT-small
`
./slurm_scripts/submit.sh scripts/run_fineweb1B_adam.sh param_configs/adamw.json small_adamw 2 
`

### Rho, mu grid search
`
./slurm_scripts/submit.sh  scripts/run_fineweb1B_adam_fista.sh param_configs/attn_fista_rho_mu_sweep.json small_fista  16
`
`
./slurm_scripts/submit.sh  scripts/run_fineweb1B_adam_fista.sh param_configs/attn_fista_rho_mu_sweep_missed.json small_fista  2
`

`
./slurm_scripts/submit.sh  scripts/run_fineweb1B_adam_fista.sh param_configs/attn_fista_rho_mu_sweep_more_mu.json small_fista_mu  16
`
`
./slurm_scripts/submit.sh  scripts/run_fineweb1B_adam_fista.sh param_configs/attn_fista_rho_mu_sweep_all_fista_it.json small_fista  16
`
`
./slurm_scripts/submit.sh  scripts/run_fineweb1B_adam_fista.sh param_configs/attn_fista_rho_mu_sweep_20it.json small_fista_20it  16
`


### Momentum, rho and mu

`
./slurm_scripts/submit.sh  scripts/run_fineweb1B_adam_fista_momentum.sh param_configs/attn_fista_rho_mu_sweep_20it_prior_m.json sm_f_20it_pm  16
`
`
./slurm_scripts/submit.sh  scripts/run_fineweb1B_adam_fista_momentum.sh param_configs/attn_fista_rho_mu_sweep_20it_prior_mv.json sm_f_20it_pmv  16
`
`
./slurm_scripts/submit.sh  scripts/run_fineweb1B_adam_fista_momentum.sh param_configs/attn_fista_rho_mu_sweep_20it_prior_mv_extra.json sm_f_20it_pmv  16
`

### Lr sweep
`
./slurm_scripts/submit.sh  scripts/run_fineweb1B_adam_fista.sh param_configs/attn_fista_rhomu_best1_lr_sweep.json small_fista_sweep  7
`
`
./slurm_scripts/submit.sh  scripts/run_fineweb1B_adam_fista.sh param_configs/attn_fista_rhomu_best2_lr_sweep.json small_fista_sweep  7
`
`
./slurm_scripts/submit.sh  scripts/run_fineweb1B_adam_fista.sh param_configs/attn_fista_rhomu_best3_lr_sweep.json small_fista_sweep3  8
`
`
./slurm_scripts/submit.sh  scripts/run_fineweb1B_adam_fista.sh param_configs/attn_fista_rhomu_best4_lr_sweep.json sm_f_sw4  8
`
`
./slurm_scripts/submit.sh  scripts/run_fineweb1B_adam_fista.sh param_configs/attn_fista_rhomu_best5_lr_sweep.json sm_f_sw5  8
`

`
./slurm_scripts/submit.sh  scripts/run_fineweb1B_adam_fista.sh param_configs/attn_fista_rhomu_best3_lr_1.json small_fista_sweep3  1
`
`
./slurm_scripts/submit.sh  scripts/run_fineweb1B_adam_fista.sh param_configs/attn_fista_rhomu_best4_lr_1.json sm_f_sw4  1
`
`
./slurm_scripts/submit.sh  scripts/run_fineweb1B_adam_fista.sh param_configs/attn_fista_rhomu_best5_lr_1.json sm_f_sw5  1
`
`
./slurm_scripts/submit.sh  scripts/run_fineweb1B_adam_fista.sh param_configs/attn_fista_rhomu_best5p5_lr_sweep.json sm_f_sw5p5  8
`


#### Lr sweep with momentum
`
./slurm_scripts/submit.sh  scripts/run_fineweb1B_adam_fista_momentum.sh param_configs/attn_fista_rhomu_best6_lr_sweep.json sm_f_sw6  9
`

`
./slurm_scripts/submit.sh  scripts/run_fineweb1B_adam_fista_momentum.sh param_configs/attn_fista_rhomu_best7_lr_sweep.json sm_f_sw7  9
`
`
./slurm_scripts/submit.sh  scripts/run_fineweb1B_adam_fista_momentum.sh param_configs/attn_fista_rhomu_best8_lr_sweep.json sm_f_sw8  9
`
`
./slurm_scripts/submit.sh  scripts/run_fineweb1B_adam_fista_momentum.sh param_configs/attn_fista_rhomu_best9_lr_sweep.json sm_f_sw9  9
`

### Baselines
`
./slurm_scripts/submit.sh scripts/run_fineweb1B_adam.sh param_configs/adamw_kq.json sm_sweep_lr_baselines 16
`
`
./slurm_scripts/submit.sh scripts/run_fineweb1B_adam.sh param_configs/adamw_kq_wclip.json sm_sweep_lr_baselines 7
`
`
./slurm_scripts/submit.sh scripts/run_fineweb1B_adam.sh param_configs/adamw_kq_lr_larger.json sm_larger_lr_baselines 16
`
`
./slurm_scripts/submit.sh scripts/run_fineweb1B_adam.sh param_configs/adamw_kq_wclip.json sm_sweep_lr_wclip 11
`

### Wandb sync
`
wandb sync /mnt/ceph/users/tparshakova/wandb_offline/wandb/offline-run-*
`