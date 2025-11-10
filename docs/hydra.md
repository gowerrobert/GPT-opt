# Hydra Configuration Guide

Hydra is a lightweight framework for composing complex configurations from simple YAML files. It lets you:

- break giant config files into small, focused pieces
- override any value at runtime from the command line or Python API
- sweep over parameter combinations without writing custom scripts
- snapshot the exact configuration used in every run

The project uses Hydra to make experiments reproducible, auditable, and easy to customize. This guide gives a general overview of Hydra concepts and how we apply them in `GPT-opt`.

---

## Core Concepts

### Config Groups

Hydra organizes related configs into *groups*. Each group contains one or more options (YAML files). For example:

```
hydra_conf/
├── model/
│   ├── gpt-small.yaml
│   ├── gpt-medium.yaml
│   └── gpt-large.yaml
├── optimizer/
│   ├── adamw.yaml
│   └── dap-ns-nadam.yaml
└── training/
    ├── slim_pajama10B.yaml
    └── fineweb1B.yaml
```

Only one option per group is active at a time. The defaults are declared in `hydra_conf/config.yaml`, but you can swap any group member via the CLI:

```bash
python run_hydra.py model=gpt-medium optimizer=dap-ns-nadam
```

### Overrides

Nested fields can be overwritten with dotted keys:

```bash
python run_hydra.py \
    optimizer.optimizer_params.lr=0.0005 \
    training.training_params.max_steps=1000
```

Hydra resolves overrides after loading the base YAMLs, so changes are non-destructive—you never edit the source files for a one-off experiment.

Interpolation lets configs reference each other. Example inside a YAML file:

```yaml
checkpoint_dir: ${paths.output_dir}/${model.name}
```

The `${...}` expression is resolved at runtime after all overrides are applied.

### Multirun Sweeps

Passing `-m` (short for `--multirun`) creates a sweep over multiple values. Hydra executes the cartesian product of options and stores each run under `multirun/<timestamp>/`.

```bash
python run_hydra.py -m \
    optimizer.optimizer_params.lr=0.0003,0.001,0.003 \
    training.training_params.batch_size=16,32
```

You can combine multiruns with SLURM by generating overrides externally (as our `submit.sh` does) and letting `disBatch` dispatch tasks across GPUs.

### Config Composition & Inheritance

Hydra supports hierarchical config composition using the `defaults` list within YAML files. This allows one config to extend another:

```yaml
# hydra_conf/training/slim_pajama10B.yaml
defaults:
  - _self_
  - override hydra/job_logging: colorlog

training_params:
  dataset: slim_pajama10B
  max_steps: 2000
```

The `_self_` entry ensures the current file is processed after any inherited configs, so local fields take precedence.

---

## Why Hydra Is Useful

- **Composability:** Swap experiment components (model, optimizer, data) independently.
- **Reproducibility:** Every run captures an immutable config snapshot under `.hydra/`.
- **Auditability:** Overrides are logged along with the final merged config, making it easy to track how a run was configured.
- **Automation:** Multirun sweeps, CLI overrides, and `hydra.sweeper` plugins simplify hyper-parameter searches before wrapping them in SLURM.
- **Python Integration:** Configs load as OmegaConf objects, enabling type-safe access (`cfg.optimizer.optimizer_params.lr`) inside Python code.

---

## Workflow in `GPT-opt`

1. **Select base configs** using CLI group overrides, e.g. `model=gpt-small training=slim_pajama10B`.
2. **Apply fine-grained tweaks** with dotted overrides for individual fields.
3. **Run locally** for quick iteration or use `-m` to test small sweeps.
4. **Scale up** by calling `slurm_scripts/submit.sh`, which translates parameter JSON files into Hydra override strings.
5. **Inspect outputs** in `outputs/<model>/<dataset>/<optimizer>/<run_name>/`, where Hydra stores `config.yaml`, `overrides.yaml`, and `hydra.yaml`.

---

## Best Practices

- Keep configs declarative—avoid embedding logic; put computed fields in Python.
- Use interpolation to avoid repeating directory paths or common names.
- Prefer `+logging=wandb`-style additions for optional features rather than editing defaults.
- Version control your config changes just like code changes.
- For complex sweeps, prototype with Hydra multirun locally before drafting a SLURM parameter file.

---

## Further Reading

- Hydra documentation: <https://hydra.cc/docs/intro/>
- OmegaConf documentation: <https://omegaconf.readthedocs.io/en/latest/>
- Hydra examples: <https://github.com/facebookresearch/hydra/tree/main/examples>

For project-specific questions, check the main `README.md` or explore the configs under `hydra_conf/`.

