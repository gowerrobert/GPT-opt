#!/bin/bash

lr=${1:-0.0001}
wd=${2:-0.0}

python run_hydra.py \
    optimizer=dap-ns-nadam \
    data=fineweb1B \
    optimizer.optimizer_params.lr=$lr \
    optimizer.optimizer_params.weight_decay=$wd