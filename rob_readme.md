# On the cluster
### srun
```bash
srun --gpus=1 --cpus-per-gpu=8 --time=150:00:00 --partition=gpu --constraint=a100 --pty bash
module load python
```

### See current jobs
```bash
squeue --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R" --me
```