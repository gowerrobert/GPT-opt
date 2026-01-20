#!/bin/bash

# lr=${1:-0.001}
# n_embd=${2:-768}
# enable_mup_with_base_n_embd=${3:-null}

python run_hydra.py --multirun \
    model=gpt-tiny \
    training=shakespeare \
    data=shakespeare \
    logging=default \
    optimizer=adamw \
    'paths.run_name=muP_sweep' \
    'optimizer.optimizer_params.lr=1e-5,1e-4,1e-3,1e-2,1e-1' \
    'model.config.n_embd=768,1536,3072' \
    +model.config.enable_mup_with_base_n_embd=null,768 \
    # optimizer=attn_fista_adamw \
    # optimizer.optimizer_params.rho_over_lr=$rho_over_lr \
    # optimizer.optimizer_params.attn_max_iter=$attn_max_iter \
    # optimizer.optimizer_params.warm_start=$warm_start \
    # optimizer.optimizer_params.momentum=$momentum \
    # optimizer.optimizer_params.mu_frac=$mu_frac
