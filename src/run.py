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
        do_class_weights=run.config.do_class_weights,
        optimizer=run.config.optimizer,
        learning_rate=run.config.learning_rate,
        weight_decay=run.config.weight_decay,
        criterion=run.config.criterion,
        scheduler=run.config.scheduler,
        epochs=run.config.epochs,
        batch_size=run.config.batch_size,
        solarize=0,
        saturation=0,
        contrast=0,
        brightness=0,
        sharpness=0,
        hue=0,
        posterization=0,
        rotation=0,
        erasing=0,
        affine=0,
        crop=0,
        gaussian_noise=0,
    )
    # Transforms
    transforms_creator = TransformsCreator(parameter_storage)
    transforms_creator.create_transforms()
    # Load the dataset
    dataset_loader = DatasetLoader(parameter_storage, transforms=transforms_creator)
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
            "size": (400, 400),
            "optimizer": "adam",
            "criterion": "cross_entropy",
            "scheduler": "plateau",
            "learning_rate": 1e-4,
            "weight_decay": 0,
            "epochs": 300,
            "batch_size": 32,
            "do_class_weights": True,
            "do_oversampling": False,
        },
    )
    main(run)
    run.finish()
