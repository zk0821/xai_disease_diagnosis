#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4GB
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/grad_cam.out
#SBATCH --error=logs/grad_cam.err
#SBATCH --job-name="GradCam"
#SBATCH --exclude=gwn[01-03],gwn10,gwn[04-06]

srun singularity exec --nv containers/container.sif python src/grad_cam.py