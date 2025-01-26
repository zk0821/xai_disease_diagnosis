singularity build containers/container.sif docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
singularity exec containers/container.sif pip install install torch torchvision torchaudio
singularity exec containers/container.sif pip install requests
singularity exec containers/container.sif pip install opencv-python-headless
singularity exec containers/container.sif pip install matplotlib
singularity exec containers/container.sif pip install seaborn
singularity exec containers/container.sif pip install scikit-image
singularity exec containers/container.sif pip install scikit-learn
singularity exec containers/container.sif pip install wandb
singularity exec containers/container.sif pip install python-dotenv