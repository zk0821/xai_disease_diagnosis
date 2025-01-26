#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --output=logs/test.out
#SBATCH --error=logs/test.err
#SBATCH --job-name="Test"
export SSL_CERT_FILE=containers/cacert.pem

srun singularity exec --nv containers/container.sif python src/test.py