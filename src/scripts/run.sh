#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --output=logs/run.out
#SBATCH --error=logs/run.err
#SBATCH --job-name="Run"
export SSL_CERT_FILE=containers/cacert.pem

srun singularity exec --nv containers/container.sif python src/run.py