import torch
from torchvision.models import vgg16, VGG16_Weights
from data.transforms_creator import TransformsCreator
from data.dataset_loader import DatasetLoader
from data.data_loader_creator import DataLoaderCreator
from models.lrp.lrp import LRPModel
from models.lrp.lrp_visualize import plot_relevance_scores
from utils.parameter_storage import ParameterStorage


def main():
    parameter_storage = ParameterStorage(
        model_architecture="resnet",
        model_type="resnet50",
        dataset="HAM_10000",
        size=(224, 224),
        do_oversampling=False,
        do_class_weights=False,
        optimizer="adam",
        learning_rate=0.001,
        weight_decay=0,
        criterion="loss",
        scheduler="none",
        epochs=10,
        batch_size=1,
        name="x",
        solarize=0,
        saturation=(0, 1),
        contrast=(0, 1),
        brightness=(0, 1),
        sharpness=0,
        hue=0,
        posterization=0,
        rotation=0,
        erasing=0,
        affine=0,
        crop=(0, 1),
        gaussian_noise=0,
    )
    device = torch.device("cuda")
    # Transforms
    transforms_creator = TransformsCreator(parameter_storage)
    transforms_creator.create_transforms()
    # Load the dataset
    dataset_loader = DatasetLoader(parameter_storage, transforms=transforms_creator)
    # Create the data loaders
    data_loader_creator = DataLoaderCreator(parameter_storage, dataset_loader)
    data_loader_creator.create_dataloaders()
    model = vgg16(weights=VGG16_Weights.DEFAULT)
    model.to(device)
    lrp_model = LRPModel(model, top_k=0.02)
    for i, (x, y) in enumerate(data_loader_creator.test_dataloader):
        x = x.to(device)
        r = lrp_model.forward(x)
        plot_relevance_scores(x=x, r=r, name=str(i))


if __name__ == "__main__":
    main()
