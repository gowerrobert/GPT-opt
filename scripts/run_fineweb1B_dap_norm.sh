#!/bin/bash

lr=${1:-0.0003}
wd=${2:-0.1}

python run_hydra.py \
    optimizer=dap-ns-nadam \
    data=fineweb1B \
    paths=upnorm \
    "optimizer.optimizer_params.lr=${lr}" \
    "optimizer.optimizer_params.weight_decay=${wd}" \
    "optimizer.optimizer_params.update_norm=True"