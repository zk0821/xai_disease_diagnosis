import os

from utils.parameter_storage import ParameterStorage
from data.transforms_creator import TransformsCreator
from data.dataset_loader import DatasetLoader
from data.data_loader_creator import DataLoaderCreator
from evaluation.evaluator import Evaluator
from models.model_handler import ModelHandler
import wandb
from dotenv import load_dotenv


def main(run):
    # Define all necessary parameters
    parameter_storage = ParameterStorage(
        name=run.name,
        model_architecture=run.config.model_architecture,
        model_type=run.config.model_type,
        dataset=run.config.dataset,
        size=run.config.size,
        do_oversampling=run.config.do_oversampling,
        class_weights=run.config.class_weights,
        optimizer=run.config.optimizer,
        learning_rate=run.config.learning_rate,
        weight_decay=run.config.weight_decay,
        criterion=run.config.criterion,
        scheduler=run.config.scheduler,
        epochs=run.config.epochs,
        batch_size=run.config.batch_size,
        solarize=run.config.solarize,
        saturation=run.config.saturation,
        contrast=run.config.contrast,
        brightness=run.config.brightness,
        sharpness=run.config.sharpness,
        hue=run.config.hue,
        posterization=run.config.posterization,
        rotation=run.config.rotation,
        erasing=run.config.erasing,
        affine=run.config.affine,
        crop=run.config.crop,
        gaussian_noise=run.config.gaussian_noise,
        focal_loss_gamma=run.config.focal_loss_gamma,
        class_balance_beta=run.config.class_balance_beta,
        augmentation_probability=run.config.augmentation_probability,
        validation_split=run.config.validation_split,
        augmentation_policy=run.config.augmentation_policy,
        augmentation_magnitude=run.config.augmentation_magnitude,
    )
    # Transforms
    transforms_creator = TransformsCreator(parameter_storage)
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
    model_handler.train_model()
    model_handler.test_model()


if __name__ == "__main__":
    load_dotenv()
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT")
    wandb.login(key=WANDB_API_KEY, relogin=True)
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        config={
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
            "do_oversampling": False,
            "solarize": 128,
            "saturation": (0.8, 1.2),
            "contrast": (0.8, 1.2),
            "brightness": (0.8, 1.2),
            "sharpness": 1,
            "hue": 0.0,
            "posterization": 5,
            "rotation": 30,
            "erasing": 0.2,
            "affine": 0.1,
            "crop": (0.7, 1.0),
            "gaussian_noise": 0.0,
            "focal_loss_gamma": 2,
            "class_balance_beta": 0.99,
            "augmentation_probability": 0.7,
            "validation_split": 0.2,
            "augmentation_policy": "v1_0",
            "augmentation_magnitude": 5,
        },
    )
    main(run)
    run.finish()
