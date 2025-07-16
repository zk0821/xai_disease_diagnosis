singularity build containers/singularity_container.sif docker://pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
singularity exec containers/singularity_container.sif pip install install torch torchvision torchaudio
singularity exec containers/singularity_container.sif pip install scikit-image
singularity exec containers/singularity_container.sif pip install scikit-learn
singularity exec containers/singularity_container.sif pip install wandb
singularity exec containers/singularity_container.sif pip install python-dotenv
singularity exec containers/singularity_container.sif pip install faiss-cpu
singularity exec containers/singularity_container.sif pip install pytorch-ignite