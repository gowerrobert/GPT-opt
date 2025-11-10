#!/bin/bash

lr=${1:-0.0003}
rcond=${2:-0.01}

python run_hydra.py \
    optimizer=dap-ns-nadam \
    data=fineweb1B \
    paths=rcond-sweep \
    optimizer.optimizer_params.lr=$lr \
    optimizer.optimizer_params.rcond=$rcond \
    optimizer.optimizer_params.accelerated=$rcond