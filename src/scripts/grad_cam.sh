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
#SBATCH --exclude=gwn[01-07],gwn[08-10]

srun singularity exec --nv containers/singularity_container.sif python src/grad_cam.py

#srun \
#    --container-image=./containers/gradcam_enroot_container.sqfs \
#    --container-mounts ${PWD}:${PWD} \
#    --container-workdir ${PWD} \
#    bash -c "python src/grad_cam.py"