#!/bin/bash

lr=${1:-0.0001}
wd=${2:-0.0}
model=${3:-gpt-small}

python run_hydra.py \
    model=$model \
    optimizer=adamw \
    data=fineweb1B \
    optimizer.optimizer_params.lr=$lr \
    optimizer.optimizer_params.weight_decay=$wd 