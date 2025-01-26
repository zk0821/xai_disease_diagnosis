from utils.parameter_storage import ParameterStorage
from data.transforms_creator import TransformsCreator
from data.dataset_loader import DatasetLoader
from data.data_loader_creator import DataLoaderCreator
import torch


def main():
    # Define all necessary parameters
    parameter_storage = ParameterStorage(
        model_architecture="efficient_net",
        model_type="b5",
        dataset="HAM_10000",
        size=(456, 456),
        do_oversampling=False,
        do_loss_weights=False,
        optimizer="sgd",
        learning_rate=0.001,
        weight_decay=0,
        criterion="cross_entropy",
        scheduler="step",
        epochs=100,
        batch_size=16,
    )
    # Transforms
    transforms_creator = TransformsCreator(parameter_storage)
    transforms_creator.create_transforms()
    # Load the dataset
    dataset_loader = DatasetLoader(parameter_storage, transforms=transforms_creator)
    # Create the data loaders
    data_loader_creator = DataLoaderCreator(parameter_storage, dataset_loader)
    data_loader_creator.create_dataloaders()
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    for image, _ in data_loader_creator.train_dataloader:
        psum += image.sum(axis=[0, 2, 3])
        psum_sq += (image**2).sum(axis=[0, 2, 3])
    count = len(dataset_loader.full_train_dataframe.get_dataframe()) * 456 * 456
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean**2)
    total_std = torch.sqrt(total_var)
    print("Mean: ", total_mean.numpy())
    print("STD: ", total_std.numpy())


if __name__ == "__main__":
    main()
