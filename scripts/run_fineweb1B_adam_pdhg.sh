#!/bin/bash

lr=${1:-0.001}
max_norm_tr=${2:-0.0001}

python run_hydra.py \
    model=gpt-small-default \
    optimizer=attn_pd_adamw \
    data=fineweb1B \
    optimizer.optimizer_params.lr=$lr \
    optimizer.optimizer_params.max_norm_tr=$max_norm_tr 