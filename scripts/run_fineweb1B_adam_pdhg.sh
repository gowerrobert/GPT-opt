#!/bin/bash

lr=${1:-0.001}
rho_over_lr=${2:-1}

python run_hydra.py \
    model=gpt-small-default \
    optimizer=attn_pd_adamw \
    data=fineweb1B \
    optimizer.optimizer_params.lr=$lr \
    optimizer.optimizer_params.rho_over_lr=$rho_over_lr 