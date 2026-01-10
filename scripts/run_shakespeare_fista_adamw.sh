#!/bin/bash

lr=${1:-0.001}
rho_over_lr=${2:-10}
attn_max_iter=${3:-100}
warm_start=${4:-false}
momentum=${5:-false}
mu_frac=${6:-0.1}

python run_hydra.py \
    model=gpt-tiny \
    optimizer=attn_fista_adamw \
    data=shakespeare \
    training=shakespeare \
    optimizer.optimizer_params.lr=$lr \
    optimizer.optimizer_params.rho_over_lr=$rho_over_lr \
    optimizer.optimizer_params.attn_max_iter=$attn_max_iter \
    optimizer.optimizer_params.warm_start=$warm_start \
    optimizer.optimizer_params.momentum=$momentum \
    optimizer.optimizer_params.mu_frac=$mu_frac