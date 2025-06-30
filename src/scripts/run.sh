#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4GB
#SBATCH --partition=dev
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/run.out
#SBATCH --error=logs/run.err
#SBATCH --job-name="Run"
export SSL_CERT_FILE=containers/cacert.pem

#srun singularity exec --nv containers/container.sif python src/run.py
srun \
    --container-image=./containers/container.sqfs \
    --container-mounts ${PWD}:${PWD} \
    --container-workdir ${PWD} \
    bash -c "pip install --upgrade numpy==1.25.0 opencv-python==4.6.0.66; python src/run.py"