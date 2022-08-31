#!/bin/sh
#SBATCH -N 1
#SBATCH --partition=pall
#SBATCH -t 1:00:00
#SBATCH --export=ALL
#SBATCH --output=./slurm_logs/ipa_%x-%j.out

source ~/.bashrc
conda activate py37

cd ../
srun python -u main.py \
          --in_dataset $1 \
          --out_dataset $2  \
          --nn $3 \
          --log_path $4 \
          --cfg_path reproducibility/reproduce_config.yml

cd reproducibility
