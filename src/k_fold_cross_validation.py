import torch
import os
import wandb

from sklearn.model_selection import StratifiedKFold
from utils.parameter_storage import ParameterStorage
from data.chest_dataset import ChestDataset
from data.isic_dataset import ISICDataset
from data.dataset_loader import DatasetLoader
from data.data_loader_creator import DataLoaderCreator
from evaluation.evaluator import Evaluator
from models.model_handler import ModelHandler
from dotenv import load_dotenv

# random seed
import torch
import random
import numpy as np


def main(run):
    validation_results = []
    results = []
    parameter_storage = ParameterStorage(
        name=run.name,
        model_architecture=run.config.model_architecture,
        model_type=run.config.model_type,
        dataset=run.config.dataset,
        size=run.config.size,
        class_weights=run.config.class_weights,
        weight_strategy=run.config.weight_strategy,
        optimizer=run.config.optimizer,
        learning_rate=run.config.learning_rate,
        weight_decay=run.config.weight_decay,
        criterion=run.config.criterion,
        scheduler=run.config.scheduler,
        model_checkpoint=run.config.model_checkpoint,
        early_stoppage=run.config.early_stoppage,
        epochs=run.config.epochs,
        batch_size=run.config.batch_size,
        focal_loss_gamma=run.config.focal_loss_gamma,
        train_augmentation_policy=run.config.train_augmentation_policy,
        train_augmentation_probability=run.config.train_augmentation_probability,
        train_augmentation_magnitude=run.config.train_augmentation_magnitude,
        test_augmentation_policy=run.config.test_augmentation_policy,
        random_seed=run.config.random_seed,
    )
    # Set the random seeds for reproducibility
    torch.manual_seed(parameter_storage.random_seed)
    torch.cuda.manual_seed(parameter_storage.random_seed)
    torch.cuda.manual_seed_all(parameter_storage.random_seed)
    random.seed(parameter_storage.random_seed)
    np.random.seed(parameter_storage.random_seed)
    # Make GPUs deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Load the dataset
    dataset_loader = DatasetLoader(parameter_storage)
    # Create K-Folds
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=parameter_storage.random_seed)
    for fold, (train_ids, validation_ids) in enumerate(
        kfold.split(
            dataset_loader.full_train_dataframe.get_dataframe(),
            dataset_loader.full_train_dataframe.get_dataframe()["type"],
        )
    ):
        print(f"Fold {fold}")
        print("-------------------")
        # Set the name
        parameter_storage.name = f"{parameter_storage.name}-fold-{fold}"
        # Create the k fold datasets
        train_dataframe = dataset_loader.full_train_dataframe.get_dataframe().loc[train_ids]
        if parameter_storage.dataset == "ChestXRAY":
            dataset_loader.train_dataset = ChestDataset(
                path=dataset_loader.full_train_dataframe.path,
                dataframe=train_dataframe,
                policy=parameter_storage.train_augmentation_policy,
            )
            validation_dataframe = dataset_loader.full_train_dataframe.get_dataframe().loc[validation_ids]
            dataset_loader.validation_dataset = ChestDataset(
                path=dataset_loader.full_train_dataframe.path,
                dataframe=validation_dataframe,
                policy=parameter_storage.test_augmentation_policy,
            )
        else:
            dataset_loader.train_dataset = ISICDataset(
                path=dataset_loader.full_train_dataframe.path,
                dataframe=train_dataframe,
                policy=parameter_storage.train_augmentation_policy,
            )
            validation_dataframe = dataset_loader.full_train_dataframe.get_dataframe().loc[validation_ids]
            dataset_loader.validation_dataset = ISICDataset(
                path=dataset_loader.full_train_dataframe.path,
                dataframe=validation_dataframe,
                policy=parameter_storage.test_augmentation_policy,
            )
        # Create the data loaders
        data_loader_creator = DataLoaderCreator(parameter_storage, dataset_loader)
        data_loader_creator.create_dataloaders()
        # Create evaluator
        evaluator = Evaluator(dataset_loader.classes)
        # Create the model
        model_handler = ModelHandler(parameter_storage, evaluator)
        model_handler.prepare_model(dataset_loader, data_loader_creator)
        model_handler.train_model(log_wandb=True, fold=f"kfold-{fold}")
        model_handler.test_model(log_wandb=True, fold=f"kfold-{fold}")
        # Record metrics
        validation_balanced_accuracy = evaluator.validation_evaluator.balanced_accuracy()
        validation_results.append(validation_balanced_accuracy)
        test_balanced_accuracy = evaluator.test_evaluator.balanced_accuracy()
        results.append(test_balanced_accuracy)

    # Validation Mean, Std, Variance
    mean = np.mean(validation_results)
    std = np.std(validation_results)
    var = np.var(validation_results)
    wandb.log({"kfold/validation/mean": mean, "kfold/validation/std": std, "kfold/validation/var": var})

    # Test Mean, Std, Variance
    mean = np.mean(results)
    std = np.std(results)
    var = np.var(results)
    wandb.log({"kfold/test/mean": mean, "kfold/test/std": std, "kfold/test/var": var})


if __name__ == "__main__":
    load_dotenv()
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT")
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        config={
            "model_architecture": "ensemble",
            "model_type": "all",
            "dataset": "ChestXRAY",
            "size": (224, 224),
            "optimizer": "adam",
            "criterion": "ldam",
            "scheduler": "plateau",
            "model_checkpoint": True,
            "early_stoppage": False,
            "learning_rate": 0.03,
            "weight_decay": 0,
            "epochs": 50,
            "batch_size": 32,
            "class_weights": "balanced",
            "weight_strategy": "deferred",
            "focal_loss_gamma": 2,
            "train_augmentation_policy": "v1_0",
            "train_augmentation_probability": 0.7,
            "train_augmentation_magnitude": 5,
            "test_augmentation_policy": "multi_crop",
            "random_seed": 2025,
        },
    )
    main(run)
    run.finish()
