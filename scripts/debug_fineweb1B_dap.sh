#!/bin/bash

lr=${1:-0.0003}
wd=${2:-0.0}

python run_hydra.py \
    training.training_params.compile=False \
    training.training_params.batch_size=8 \
    optimizer.optimizer_params.debug_timing=True \
    logging=default \
    optimizer=dap-ns-nadam-accel \
    data=fineweb1B \
    "optimizer.optimizer_params.lr=${lr}" \
    "optimizer.optimizer_params.weight_decay=${wd}"