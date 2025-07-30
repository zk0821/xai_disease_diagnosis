# EXplainable Artificial Intelligence for Skin Lesion Recognition

This project explores various machine learning techniques for skin lesion recognition and builds upon them with explainable AI methods.

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Getting Started](#getting-started)

## Repository Structure

```

├── containers         <- Singularity containers
│
├── data               <- Data used for model training, validation and testing
│
├── logs               <- Logs for scripts
│
├── models             <- Trained and serialized models (empty for now)
│
├── notebooks          <- Jupyter notebooks (empty for now)
│
├── reports            <- Generated analysis as PDF, LaTeX.
│   └── figures        <- Generated graphics and figures to be used in reporting
│   └── drawio         <- Draw.io files for graphics
│
├── requirements.txt   <- The requirements file for reproducing the environment
│
├── src                <- Source code for use in this project
    ├── data           <- Source code for handling data
    ├── evaluation     <- Source code for handling evaluation
    ├── models         <- Models source files
    ├── scripts        <- Scripts source files (for use with SLURM)
    └── utils          <- Utility source files
├── .gitignore         <- Files for git to ignore
├── LICENSE            <- Open-source license
├── Makefile           <- Makefile
└── README.md          <- The top-level README
```

## Getting Started
This code has been written and tested for use in SLURM systems.

### Preparing the Datasets

#### ISIC 2018 a.k.a. HAM10000
Head to https://challenge.isic-archive.com/data/#2018

Download **Training Data** and **Test Data** for Task 3 (only images and groundtruth data is required).

Organize the dataset as shown below:
```
├── data
    └── ham10000
        ├── train
            ├── images
            └── groundtruth.csv
        └── test
            ├── images
            └── groundtruth.csv
```

### Setting up the Container
Before we can run the scripts we need to prepare the environment where these scripts will be run. For this we can choose between two options:
- Singularity Container
- Enroot Container

We will be using a Pytorch Container image as our base.

To create the container please run the following command:
- Singularity Container
```
make singularity_container
```
- Enroot Container
```
make enroot_container
```

This should create a **container.sif** file inside **containers** folder.

### Environmental variables in .env
In order to track our experiments we are using Weights & Biases (wandb.ai). Most of the scripts also report the results there and expect the following environmental variables to be set:
```
WANDB_API_KEY={wandb api key}
WANDB_ENTITY={username}
WANDB_PROJECT={name of the project}
```

### Adjust File Structure Accordingly
Before training models we need to adjust the file structure of the project, more specifically, the **models** folder to have the necessary sub-folders for our experiments.

For example we want to run an experiment with CNN architecture **Efficient Net** of type **B2**. This means the following file structure must exist.
```
├── models
    └── efficient_net
        └── b2
```
This will allow the model weights to be saved in the appropriate folder.

Please refer to src/models/model_handler and src/models/cnn/ for available model architectures and types available for that specific architecture.

### Running our first script
At this point we are ready to schedule our first job.

Scripts can be run using `make` command. The `make` command will execute a similarly named script, usually placed in `src/scripts`.

A modification is necessary when running on:
- Singularity
```
srun singularity exec --nv containers/singularity_container.sif python src/<script>.py
```
- Enroot
```
srun \
    --container-image=./containers/gradcam_enroot_container.sqfs \
    --container-mounts ${PWD}:${PWD} \
    --container-workdir ${PWD} \
    bash -c "python src/<script>.py"
```

In this example we will be scheduling a simple **Run** job, which will train a CNN or Transformer using the parameters defined inside the file. After the training is complete the model is evaluated on the test dataset.

To execute the **Run** script please run the following command:
```
make run
```

This will schedule a SLURM job and you can track the progress using Weights & Biases, or observe run.out and run.err files inside **logs** folder.

### GradCAM

For now GradCAM is setup to use Ensemble model:
- EfficientNet V2
- ConvNeXt
- SwinTransformer

The GradCAM is composed of an average of the final feature layer in each of the models comprising the ensemble.

To run GradCAM on an image run:
```
make grad_cam
```

GradCAM will be performed on the image saved in variable `image_name` in `src/grad_cam.py`. The results are saved to folder `grad_cam/<image_name>`.


### Overview of Scripts
- cbir: Content Based Image Retrieval; loads an existing model and performs CBIR on an example image (XAI: explanations with examples)
- grad_cam: Grad-CAM visualizations; loads the gradcam model (currently only ResNet supported) and performs gradient visualizations of an example image (XAI: visual explanations)
- k_fold_cross_validation: loads an existing model and performs k-fold cross validation
- lr_finder: attempts to find the "optimal" learning rate for a model architecture using idea from paper "Cyclical Learning Rates for Training Neural Networks"
- lrp: Layer-wise Relevance Propagation -> performs LRP visualization on an example image (currently only supports VGG-16)
- mean_std: used to calculate the mean and standard deviation of training dataset for normalization
- run: single training and evaluation cycle
- sweep: Weights & Biases sweep (multiple training and evaluation cycles, used for hyperparameter optimization)
- test: loads an existing model and performs evaluation on test data