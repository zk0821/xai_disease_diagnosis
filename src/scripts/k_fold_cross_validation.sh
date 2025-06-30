#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4GB
#SBATCH --partition=dev
#SBATCH --gpus=A100_80GB:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/k_fold.out
#SBATCH --error=logs/k_fold.err
#SBATCH --job-name="K Fold"
export SSL_CERT_FILE=containers/cacert.pem

#srun singularity exec --nv containers/container.sif python src/k_fold_cross_validation.py
srun \
    --container-image=./containers/container.sqfs \
    --container-mounts ${PWD}:${PWD} \
    --container-workdir ${PWD} \
    bash -c "pip install --upgrade numpy==1.25.0 opencv-python==4.6.0.66; python src/k_fold_cross_validation.py"