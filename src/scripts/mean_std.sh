#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/Mean.out
#SBATCH --error=logs/Mean.err
#SBATCH --job-name="Mean"

srun singularity exec --nv containers/container.sif python src/mean_std.py