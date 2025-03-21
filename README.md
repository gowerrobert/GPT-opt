# Matsign

Small package for testing matrix sign methods on the weights of GPT models from the Transformers library

To start, setup up the virtual enviroment and install dependencies by running
```bash
 ./setup_env.sh
```
Otherwise you can do it manually as follows:

### Create a virtual environment and activate:
```bash
python3.9 -m venv matsign
source matsign/bin/activate
python3.9 -m pip install -e .
```

### Run Example:
```bash
python3.9 run.py --config configs/gpt-med.yaml
```

### Plot Results:
```bash
module load texlive
python3.9 plot.py --config configs/gpt-med.yaml
```

# On the cluster

### srun
```bash
srun --gpus=1 --cpus-per-gpu=8 --time=150:00:00 --partition=gpu --constraint=a100 --pty bash
module load python
```

### Or using Slurm:
```bash
./submit.sh configs/gpt-med.yaml
```

### See current jobs
```bash
squeue --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R" --me
```

### Debugging
Check the version of torch
python3.9 -c "import torch; print(torch.version.cuda)"

Check the system version of CUDA
nvcc --version

Check if GPU is visible
nvidia-smi