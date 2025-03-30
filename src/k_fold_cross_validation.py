import torch
from sklearn.model_selection import KFold
from utils.parameter_storage import ParameterStorage
from data.transforms_creator import TransformsCreator
from data.dataset_loader import DatasetLoader
from data.data_loader_creator import DataLoaderCreator
from evaluation.evaluator import Evaluator
from models.model_handler import ModelHandler


def main():
    torch.manual_seed(42)
    results = {}
    parameter_storage = ParameterStorage(
        name="ethereal-sweep-50",
        model_architecture="efficient_net",
        model_type="b2",
        dataset="HAM_10000",
        size=(400, 400),
        do_oversampling=False,
        do_class_weights=True,
        optimizer="adam",
        learning_rate=0.00043833,
        weight_decay=0.00059288,
        criterion="cross_entropy",
        scheduler="plateau",
        epochs=300,
        batch_size=32,
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
        crop=(0.7, 1),
        gaussian_noise=0.0,
    )
    # Transforms
    transforms_creator = TransformsCreator(parameter_storage)
    transforms_creator.create_transforms()
    # Load the dataset
    dataset_loader = DatasetLoader(parameter_storage, transforms=transforms_creator)
    # Create K-Folds
    kfold = KFold(n_splits=5, shuffle=True)
    for fold, (train_ids, validation_ids) in enumerate(kfold.split(dataset_loader.full_train_dataset)):
        print(f"Fold {fold}")
        print("-------------------")
        # Set the name
        parameter_storage.name = f"ethereal-sweep-50-fold-{fold}"
        # Create the data loaders
        data_loader_creator = DataLoaderCreator(parameter_storage, dataset_loader)
        data_loader_creator.create_dataloaders_from_ids(train_ids, validation_ids)
        # Create evaluator
        evaluator = Evaluator(dataset_loader.classes)
        # Create the model
        model_handler = ModelHandler(parameter_storage, evaluator)
        model_handler.prepare_model(dataset_loader, data_loader_creator)
        model_handler.train_model(log_wandb=False)
        model_handler.test_model(log_wandb=False)


if __name__ == "__main__":
    main()
