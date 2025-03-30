#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/cbir.out
#SBATCH --error=logs/cbir.err
#SBATCH --job-name="CBIR"

srun singularity exec --nv containers/container.sif python src/cbir.py