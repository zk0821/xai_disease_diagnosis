singularity build containers/container_old.sif docker://pytorch/pytorch:2.5.0-cuda11.8-cudnn9-runtime
singularity exec containers/container_old.sif pip install install torch torchvision torchaudio
singularity exec containers/container_old.sif pip install scikit-image
singularity exec containers/container_old.sif pip install scikit-learn
singularity exec containers/container_old.sif pip install wandb
singularity exec containers/container_old.sif pip install python-dotenv
singularity exec containers/container_old.sif pip install faiss-cpu
singularity exec containers/container_old.sif pip install pytorch-ignite