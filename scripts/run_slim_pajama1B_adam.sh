#!/bin/bash

lr=${1:-0.0001}
wd=${2:-0.0}
opt=${3:-adamw}

python run_hydra.py \
    optimizer=$opt \
    data=slim_pajama1B \
    optimizer.optimizer_params.lr=$lr \
    optimizer.optimizer_params.weight_decay=$wd