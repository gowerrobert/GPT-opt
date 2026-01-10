#!/bin/bash

lr=${1:-0.001}
rho_over_lr=${2:-10}
attn_max_iter=${3:-100}
diag_scaling=${4:-false}
enable_restart=${5:-false}
reflected_halpern=${6:-false}
warm_start=${7:-false}

python run_hydra.py \
    model=gpt-tiny \
    optimizer=attn_rehpdhg_adamw \
    data=shakespeare \
    training=shakespeare \
    optimizer.optimizer_params.lr=$lr \
    optimizer.optimizer_params.rho_over_lr=$rho_over_lr \
    optimizer.optimizer_params.attn_max_iter=$attn_max_iter \
    optimizer.optimizer_params.diag_scaling=$diag_scaling \
    optimizer.optimizer_params.enable_restart=$enable_restart \
    optimizer.optimizer_params.reflected_halpern=$reflected_halpern \
    optimizer.optimizer_params.warm_start=$warm_start


