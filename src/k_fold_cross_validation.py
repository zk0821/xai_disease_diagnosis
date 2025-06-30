import torch
import os
import wandb

from sklearn.model_selection import StratifiedKFold
from utils.parameter_storage import ParameterStorage
from data.transforms_creator import TransformsCreator
from data.ham10000_dataset import HAM10000Dataset
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
    results = []
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
        random_seed=run.config.random_seed
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
    # Transforms
    #transforms_creator = TransformsCreator(parameter_storage)
    #transforms_creator.create_transforms()
    # Load the dataset
    dataset_loader = DatasetLoader(parameter_storage)
    # Create K-Folds
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=parameter_storage.random_seed)
    for fold, (train_ids, validation_ids) in enumerate(kfold.split(
        dataset_loader.full_train_dataframe.get_dataframe(),
        dataset_loader.full_train_dataframe.get_dataframe()["type"]
    )):
        print(f"Fold {fold}")
        print("-------------------")
        # Set the name
        parameter_storage.name = f"{parameter_storage.name}-fold-{fold}"
        # Create the k fold datasets
        train_dataframe = dataset_loader.full_train_dataframe.get_dataframe().loc[train_ids]
        dataset_loader.train_dataset = HAM10000Dataset(
            path=dataset_loader.full_train_dataframe.path,
            dataframe=train_dataframe,
            transforms=None,
            policy=parameter_storage.augmentation_policy
        )
        validation_dataframe = dataset_loader.full_train_dataframe.get_dataframe().loc[validation_ids]
        dataset_loader.validation_dataset = HAM10000Dataset(
            path=dataset_loader.full_train_dataframe.path,
            dataframe=validation_dataframe,
            transforms=None,
            policy="multi_crop"
        )
        # Create the data loaders
        data_loader_creator = DataLoaderCreator(parameter_storage, dataset_loader)
        data_loader_creator.create_dataloaders()
        #data_loader_creator.create_dataloaders_from_ids(train_ids, validation_ids)
        # Create evaluator
        evaluator = Evaluator(dataset_loader.classes)
        # Create the model
        model_handler = ModelHandler(parameter_storage, evaluator)
        model_handler.prepare_model(dataset_loader, data_loader_creator)
        model_handler.train_model(log_wandb=False)
        model_handler.test_model(log_wandb=False)
        test_balanced_accuracy = evaluator.test_evaluator.balanced_accuracy()
        results.append(test_balanced_accuracy)
        wandb.log(
            {
                f"kfold-{fold}/test_bmca": test_balanced_accuracy
            }
        )
    # Mean, Std, Variance
    mean = np.mean(results)
    std = np.std(results)
    var = np.var(results)
    wandb.log(
        {
            "kfold/mean": mean,
            "kfold/std": std,
            "kfold/var": var
        }
    )


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
            "scheduler": "multi_step",
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
            "class_balance_beta": 0.999,
            "augmentation_probability": 0.7,
            "validation_split": 0.2,
            "augmentation_policy": "v1_0",
            "augmentation_magnitude": 5,
            "random_seed": 380
        },
    )
    main(run)
    run.finish()
