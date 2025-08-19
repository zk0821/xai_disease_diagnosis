#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4GB
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/lrp.out
#SBATCH --error=logs/lrp.err
#SBATCH --job-name="LRP"
#SBATCH --exclude=gwn[01-07],gwn[08-10]
export SSL_CERT_FILE=containers/cacert.pem

srun singularity exec --nv containers/singularity_container.sif python src/lrp.py