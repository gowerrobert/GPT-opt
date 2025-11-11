
module load python
source venv/bin/activate
time time torchrun --standalone --nproc_per_node=1 run.py configs/shakespeare.yaml