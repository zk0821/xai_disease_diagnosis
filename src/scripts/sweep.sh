#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --time=3-12:00:00
#SBATCH --output=logs/sweep.out
#SBATCH --error=logs/sweep.err
#SBATCH --job-name="Sweep"
export SSL_CERT_FILE=containers/cacert.pem

srun singularity exec --nv containers/container.sif python src/sweep.py