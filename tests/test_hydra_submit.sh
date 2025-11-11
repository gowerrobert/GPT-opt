#!/bin/bash

./slurm_scripts/submit.sh \
    scripts/run_shakespeare_adam.sh \
    param_configs/adamw.json \
    test_hydra_submit \
    2