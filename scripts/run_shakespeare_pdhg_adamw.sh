#!/bin/bash

lr=${1:-0.001}
max_norm_tr=${2:-0.1}

python run_hydra.py \
    model=gpt-tiny \
    optimizer=attn_pdhg_adamw \
    data=shakespeare \
    training=shakespeare \
    optimizer.optimizer_params.lr=$lr \
    optimizer.optimizer_params.max_norm_tr=$max_norm_tr 