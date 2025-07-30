#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4GB
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/test.out
#SBATCH --error=logs/test.err
#SBATCH --job-name="Test"
#SBATCH --exclude=gwn[01-07],gwn[08-10]
export SSL_CERT_FILE=containers/cacert.pem

srun singularity exec --nv containers/singularity_container.sif python src/test.py
#srun \
#    --container-image=./containers/container.sqfs \
#    --container-mounts ${PWD}:${PWD} \
#    --container-workdir ${PWD} \
#    bash -c "pip install --upgrade numpy==1.25.0 opencv-python==4.6.0.66; python src/test.py"