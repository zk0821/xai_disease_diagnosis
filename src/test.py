import os

from utils.parameter_storage import ParameterStorage
from data.transforms_creator import TransformsCreator
from data.dataset_loader import DatasetLoader
from data.data_loader_creator import DataLoaderCreator
from evaluation.evaluator import Evaluator
from models.model_handler import ModelHandler
import wandb
from dotenv import load_dotenv


def main():
    # Define all necessary parameters
    parameter_storage = ParameterStorage(
        name=wandb.config.name,
        model_architecture=wandb.config.model_architecture,
        model_type=wandb.config.model_type,
        dataset=wandb.config.dataset,
        size=wandb.config.size,
        do_oversampling=False,
        class_weights=wandb.config.class_weights,
        optimizer=wandb.config.optimizer,
        learning_rate=wandb.config.learning_rate,
        weight_decay=wandb.config.weight_decay,
        criterion=wandb.config.criterion,
        scheduler=wandb.config.scheduler,
        epochs=wandb.config.epochs,
        batch_size=wandb.config.batch_size,
        solarize=128,
        saturation=(0.8, 1.2),
        contrast=(0.8, 1.2),
        brightness=(0.8, 1.2),
        sharpness=1,
        hue=0.0,
        posterization=5,
        rotation=30,
        erasing=0.2,
        affine=0.1,
        crop=(0.7, 1.0),
        gaussian_noise=0.0,
        focal_loss_gamma=2,
        class_balance_beta=0.999,
        augmentation_probability=0.7,
        validation_split=0.2,
        augmentation_policy="v1_0",
        augmentation_magnitude=5
    )
    # Transforms
    transforms_creator = TransformsCreator(parameter_storage, tta=True)
    transforms_creator.create_transforms()
    # Load the dataset
    dataset_loader = DatasetLoader(parameter_storage)
    # Create the data loaders
    data_loader_creator = DataLoaderCreator(parameter_storage, dataset_loader)
    data_loader_creator.create_dataloaders()
    # Create evaluator
    evaluator = Evaluator(dataset_loader.classes)
    # Create the model
    model_handler = ModelHandler(parameter_storage, evaluator)
    model_handler.prepare_model(dataset_loader, data_loader_creator)
    model_handler.test_model()


if __name__ == "__main__":
    load_dotenv()
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT")
    wandb.login(key=WANDB_API_KEY, relogin=True)
    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        config={
            "name": "soft-morning-2930",
            "model_architecture": "efficient_net",
            "model_type": "b2",
            "dataset": "HAM_10000",
            "size": (224, 224),
            "optimizer": "adam",
            "criterion": "cross_entropy",
            "scheduler": "none",
            "learning_rate": 2e-4,
            "weight_decay": 1e-4,
            "epochs": 300,
            "batch_size": 32,
            "class_weights": "reweight",
        },
    )
    main()
    wandb.finish()
