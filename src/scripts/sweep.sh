#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4GB
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/sweep.out
#SBATCH --error=logs/sweep.err
#SBATCH --job-name="Sweep"
#SBATCH --exclude=gwn[01-03],gwn10,gwn[04-06],gwn08,wn222,wn224,wn209,wn210,wn211,wn212
export SSL_CERT_FILE=containers/cacert.pem

srun singularity exec --nv containers/container.sif python src/sweep.py