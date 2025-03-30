#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/lr_finder.out
#SBATCH --error=logs/lr_finder.err
#SBATCH --job-name="LR Finder"
export SSL_CERT_FILE=containers/cacert.pem

srun singularity exec --nv containers/container.sif python src/lr_finder.py