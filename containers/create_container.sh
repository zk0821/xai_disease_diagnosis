singularity build containers/container.sif docker://pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
singularity exec containers/container.sif pip install install torch torchvision torchaudio
singularity exec containers/container.sif pip install scikit-image
singularity exec containers/container.sif pip install scikit-learn
singularity exec containers/container.sif pip install wandb
singularity exec containers/container.sif pip install python-dotenv
singularity exec containers/container.sif pip install faiss-cpu
singularity exec containers/container.sif pip install pytorch-ignite