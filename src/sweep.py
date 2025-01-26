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
    run = wandb.init()
    # Define all necessary parameters
    parameter_storage = ParameterStorage(
        name=run.name,
        model_architecture=run.config.model_architecture,
        model_type=run.config.model_type,
        dataset=run.config.dataset,
        size=run.config.size,
        do_oversampling=False,
        do_class_weights=run.config.do_class_weights,
        optimizer=run.config.optimizer,
        learning_rate=run.config.learning_rate,
        weight_decay=run.config.weight_decay,
        criterion=run.config.criterion,
        scheduler=run.config.scheduler,
        epochs=run.config.epochs,
        batch_size=run.config.batch_size,
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
    sweep_configuration = {
        "method": "bayes",
        "name": "conv-next-large-class-weights",
        "metric": {"goal": "minimize", "name": "test/loss"},
        "parameters": {
            "dataset": {"values": ["HAM_10000"]},
            "size": {"values": [(400, 400)]},
            "model_architecture": {"values": ["conv_next"]},
            "model_type": {"values": ["convnext_large"]},
            "learning_rate": {"max": 1e-3, "min": 1e-6},
            "weight_decay": {"max": 1e-3, "min": 0.0},
            "epochs": {"values": [400]},
            "optimizer": {"values": ["adam"]},
            "criterion": {"values": ["cross_entropy"]},
            "scheduler": {"values": ["plateau"]},
            "batch_size": {"values": [32]},
            "do_class_weights": {"values": [True]}
        },
    }
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
    )
    wandb.agent(sweep_id, function=main, count=50)
