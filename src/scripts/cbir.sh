#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4GB
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/cbir.out
#SBATCH --error=logs/cbir.err
#SBATCH --job-name="CBIR"
#SBATCH --exclude=gwn[01-07],gwn[08-10]

srun singularity exec --nv containers/singularity_container.sif python src/cbir.py