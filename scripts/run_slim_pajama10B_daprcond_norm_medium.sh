#!/bin/bash

lr=${1:-0.0001}
wd=${2:-0.1}
rcond=${3:-0.01}

python run_hydra.py \
    model=gpt-medium \
    paths=upnorm \
    optimizer=dap-ns-nadam \
    training=slim_pajama10B \
    data=slim_pajama10B \
    optimizer.optimizer_params.lr=$lr \
    optimizer.optimizer_params.weight_decay=$wd \
    optimizer.optimizer_params.rcond=$rcond \
    optimizer.optimizer_params.accelerated=$rcond \
    optimizer.optimizer_params.update_norm=True \
    logging.logging_params.val_step=1024 \
    logging.logging_params.log_step=1024