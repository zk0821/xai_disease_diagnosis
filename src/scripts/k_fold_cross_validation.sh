#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4GB
#SBATCH --partition=dev
#SBATCH --gpus=A100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/k_fold.out
#SBATCH --error=logs/k_fold.err
#SBATCH --job-name="K Fold"
export SSL_CERT_FILE=containers/cacert.pem

#srun singularity exec --nv containers/singularity_container.sif python src/k_fold_cross_validation.py

srun \
    --container-image=./containers/enroot_container.sqfs \
    --container-mounts ${PWD}:${PWD} \
    --container-workdir ${PWD} \
    bash -c "python src/k_fold_cross_validation.py"