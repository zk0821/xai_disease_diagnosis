#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4GB
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/run.out
#SBATCH --error=logs/run.err
#SBATCH --job-name="Run"
#SBATCH --exclude=gwn[01-03],gwn10,gwn[04-06]
export SSL_CERT_FILE=containers/cacert.pem

srun singularity exec --nv containers/container.sif python src/run.py