#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4GB
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/k_fold.out
#SBATCH --error=logs/k_fold.err
#SBATCH --job-name="K Fold"
#SBATCH --exclude=gwn01
export SSL_CERT_FILE=containers/cacert.pem

srun singularity exec --nv containers/container.sif python src/k_fold_cross_validation.py