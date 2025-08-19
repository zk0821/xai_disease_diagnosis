singularity build containers/singularity_container.sif docker://nvcr.io/nvidia/pytorch:23.08-py3
singularity exec containers/singularity_container.sif pip install install torch torchvision torchaudio
singularity exec containers/singularity_container.sif pip install scikit-image
singularity exec containers/singularity_container.sif pip install scikit-learn
singularity exec containers/singularity_container.sif pip install wandb
singularity exec containers/singularity_container.sif pip install python-dotenv
singularity exec containers/singularity_container.sif pip install faiss-cpu
singularity exec containers/singularity_container.sif pip install pytorch-ignite
singularity exec containers/singularity_container.sif pip uninstall opencv-python
singularity exec containers/singularity_container.sif pip install opencv-python-headless
singularity exec containers/singularity_container.sif pip install cvxopt
singularity exec containers/singularity_container.sif pip install grad-cam
singularity exec containers/singularity_container.sif pip install captum
singularity exec containers/singularity_container.sif pip install --upgrade numpy==1.25.0