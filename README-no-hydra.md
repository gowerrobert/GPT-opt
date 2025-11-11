# GPT-opt

Small package for testing optimization methods for training GPT models from the Transformers library

 start, setup up the virtual enviroment and install dependencies by running
```bash
 ./setup_env.sh
```


### Run Example:
```bash
python run.py configs/shakespeare.yaml
```

### Plot Results:
```bash
python plot.py configs/shakespeare.yaml
```

# On the cluster
### srun
```bash
srun --gpus=1 --cpus-per-gpu=8 --time=150:00:00 --partition=gpu --constraint=a100 --pty bash
module load python
```

### Or using Slurm:
```bash
./submit.sh configs/shakespeare.yaml
```

### See current jobs
```bash
squeue --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R" --me
```


