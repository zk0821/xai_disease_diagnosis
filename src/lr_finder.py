from data.transforms_creator import TransformsCreator
from data.dataset_loader import DatasetLoader
from data.data_loader_creator import DataLoaderCreator
from utils.parameter_storage import ParameterStorage
from models.cnn.efficientnet_model import CustomEfficientNet
from xai_skin_lesion_recognition.src.models.loss.custom_loss import FocalLoss
from sklearn.utils.class_weight import compute_class_weight

import os
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, Accuracy
from ignite.contrib.handlers import FastaiLRFinder, ProgressBar
import numpy as np


def main():
    parameter_storage = ParameterStorage(
        name="Learning Rate Finder",
        model_architecture="efficient_net",
        model_type="b2",
        dataset="HAM_10000",
        size=(400, 400),
        do_oversampling=False,
        do_class_weights=True,
        optimizer="adam",
        learning_rate=1e-3,
        weight_decay=0.0,
        criterion="cross_entropy",
        scheduler="none",
        epochs=200,
        batch_size=16,
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
        crop=(0.7, 1.0),
        gaussian_noise=0.0,
    )
    # Transforms
    transforms_creator = TransformsCreator(parameter_storage)
    transforms_creator.create_transforms()

    # Load the dataset
    dataset_loader = DatasetLoader(parameter_storage, transforms=transforms_creator)

    # Create the distributed data loaders
    data_loader_creator = DataLoaderCreator(parameter_storage, dataset_loader)
    data_loader_creator.create_dataloaders()

    # Create the model
    device = "cuda"
    efficient_net = CustomEfficientNet(type="b2", num_classes=7)
    efficient_net.to(device)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(dataset_loader.train_dataframe["type"].to_numpy()),
        y=dataset_loader.train_dataframe["type"].to_numpy(),
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    criterion = FocalLoss(alpha=class_weights, gamma=2, reduction="mean")

    optimizer = optim.Adam(efficient_net.parameters(), lr=1e-6)

    trainer = create_supervised_trainer(efficient_net, optimizer, criterion, device=device)

    ProgressBar(persist=True).attach(trainer, output_transform=lambda x: {"batch_loss": x})

    lr_finder = FastaiLRFinder()

    to_save = {"model": efficient_net, "optimizer": optimizer}
    with lr_finder.attach(trainer, to_save, diverge_th=1.5) as trainer_with_lr_finder:
        print("Running")
        trainer_with_lr_finder.run(data_loader_creator.train_dataloader)
    print("Trainer running")
    trainer.run(data_loader_creator.train_dataloader, max_epochs=20)
    evaluator = create_supervised_evaluator(
        efficient_net,
        metrics={"acc": Accuracy(), "loss": Loss(FocalLoss(alpha=class_weights, gamma=2, reduction="mean"))},
        device=device,
    )
    evaluator.run(data_loader_creator.validation_dataloader)
    print(evaluator.state.metrics)
    ax = lr_finder.plot()
    ax.figure.savefig("lr_finder_plot.png")
    print("Suggestion", lr_finder.lr_suggestion())


if __name__ == "__main__":
    main()
