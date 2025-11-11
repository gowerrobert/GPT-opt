
module load python
source venv/bin/activate
time torchrun --standalone --nproc_per_node=1 run_hydra.py model=gpt-tiny optimizer=adamw data=shakespeare training=shakespeare