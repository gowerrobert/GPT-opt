#!/bin/bash

lr=${1:-0.001}
max_norm_tr=${2:-0.1}
model=${3:-gpt-small}

python run_hydra.py \
    model=$model \
    optimizer=attn_pdhg_adamw \
    data=fineweb1B \
    optimizer.optimizer_params.lr=$lr \
    optimizer.optimizer_params.max_norm_tr=$max_norm_tr  