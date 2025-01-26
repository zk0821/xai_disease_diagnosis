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
        do_class_weights=wandb.config.do_class_weights,
        optimizer=wandb.config.optimizer,
        learning_rate=wandb.config.learning_rate,
        weight_decay=wandb.config.weight_decay,
        criterion=wandb.config.criterion,
        scheduler=wandb.config.scheduler,
        epochs=wandb.config.epochs,
        batch_size=wandb.config.batch_size,
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
            "name": "clear-oath-472",
            "model_architecture": "efficient_net",
            "model_type": "b7",
            "dataset": "HAM_10000",
            "size": (400, 400),
            "optimizer": "adam",
            "criterion": "cross_entropy",
            "scheduler": "step",
            "learning_rate": 1e-5,
            "weight_decay": 0,
            "epochs": 400,
            "batch_size": 32,
            "do_class_weights": False
        },
    )
    main()
    wandb.finish()