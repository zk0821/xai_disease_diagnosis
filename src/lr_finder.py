from data.dataset_loader import DatasetLoader
from data.data_loader_creator import DataLoaderCreator
from utils.parameter_storage import ParameterStorage
from models.cnn.resnet_model import CustomResNet
from models.cnn.densenet_model import CustomDenseNet
from models.cnn.efficientnet_model import CustomEfficientNet
from models.cnn.convnext_model import CustomConvNext
from models.transformer.vision_transformer import CustomVisionTransformer
from models.transformer.swin_transformer import CustomSwinTransformer
from models.ensemble.ensemble import EnsembleModel

import torch.nn as nn
import torch.optim as optim
from ignite.engine import create_supervised_trainer
from ignite.contrib.handlers import FastaiLRFinder, ProgressBar

import torch
import random
import numpy as np


def main():
    parameter_storage = ParameterStorage(
        name="Learning Rate Finder",
        model_architecture="ensemble",
        model_type="all",
        dataset="HAM_10000",
        size=(224, 224),
        optimizer="adam",
        learning_rate=2e-4,
        weight_decay=0,
        criterion="ldam",
        scheduler="plateau",
        epochs=70,
        batch_size=32,
        model_checkpoint=True,
        early_stoppage=False,
        class_weights="balanced",
        weight_strategy="deferred",
        focal_loss_gamma=2,
        train_augmentation_policy="v1_0",
        train_augmentation_probability=0.7,
        train_augmentation_magnitude=5,
        test_augmentation_policy="multi_crop",
        random_seed=2025,
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

    # Create the distributed data loaders
    data_loader_creator = DataLoaderCreator(parameter_storage, dataset_loader)
    data_loader_creator.create_dataloaders()

    # Create the model
    device = "cuda"
    if parameter_storage.model_architecture == "res_net":
        model = CustomResNet(type=parameter_storage.model_type, num_classes=dataset_loader.num_classes)
    elif parameter_storage.model_architecture == "dense_net":
        model = CustomDenseNet(type=parameter_storage.model_type, num_classes=dataset_loader.num_classes)
    elif parameter_storage.model_architecture == "efficient_net":
        model = CustomEfficientNet(type=parameter_storage.model_type, num_classes=dataset_loader.num_classes)
    elif parameter_storage.model_architecture == "conv_next":
        model = CustomConvNext(type=parameter_storage.model_type, num_classes=dataset_loader.num_classes)
    elif parameter_storage.model_architecture == "vision_transformer":
        model = CustomVisionTransformer(type=parameter_storage.model_type, num_classes=dataset_loader.num_classes)
    elif parameter_storage.model_architecture == "swin_transformer":
        model = CustomSwinTransformer(type=parameter_storage.model_type, num_classes=dataset_loader.num_classes)
    elif parameter_storage.model_architecture == "ensemble":
        model = EnsembleModel(dataset_loader.num_classes)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        raise RuntimeError(f"Unsupported model architecture defined: {parameter_storage.model_architecture}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

    ProgressBar(persist=True).attach(trainer, output_transform=lambda x: {"batch_loss": x})

    lr_finder = FastaiLRFinder()

    to_save = {"model": model, "optimizer": optimizer}

    with lr_finder.attach(trainer, to_save) as trainer_with_lr_finder:
        print("Running")
        trainer_with_lr_finder.run(data_loader_creator.train_dataloader)

    ax = lr_finder.plot()
    ax.figure.savefig("lr_finder_plot.png")
    print("Suggestion", lr_finder.lr_suggestion())


if __name__ == "__main__":
    main()
