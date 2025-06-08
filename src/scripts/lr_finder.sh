#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/lr_finder.out
#SBATCH --error=logs/lr_finder.err
#SBATCH --job-name="LR Finder"
#SBATCH --exclude=gwn[01-03],gwn10,gwn[04-06]
export SSL_CERT_FILE=containers/cacert.pem

srun singularity exec --nv containers/container.sif python src/lr_finder.py