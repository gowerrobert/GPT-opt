#!/bin/bash
#SBATCH -p gen                  # CPU partition
#SBATCH -c 64                   # 64 cores
#SBATCH --mem=512G              # 512 GB RAM
#SBATCH -t 04:00:00             # walltime
#SBATCH -J slim10B              # job name
#SBATCH -o slurm_logs/slim10B.%j.out  # log file

# Activate your environment
module load python
source venv/bin/activate

# Run preprocessing
python process_data.py --name fineweb10B
