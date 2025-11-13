#!/bin/bash

lr=${1:-0.0001}
muon_lr=${2:-0.0001}
wd=${3:-0.0}
name=${4:-adamw}

python run_hydra.py \
    optimizer=${name} \
    data=shakespeare \
    training=shakespeare \
    optimizer.optimizer_params.lr=$lr \
    optimizer.optimizer_params.weight_decay=$wd \
    optimizer.optimizer_params.muon_lr=$muon_lr \
    model=gpt-tiny