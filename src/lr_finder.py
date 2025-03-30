from data.transforms_creator import TransformsCreator
from data.dataset_loader import DatasetLoader
from data.data_loader_creator import DataLoaderCreator
from utils.parameter_storage import ParameterStorage
from models.cnn.efficientnet_model import CustomEfficientNet

import os
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, Accuracy
from ignite.contrib.handlers import FastaiLRFinder, ProgressBar


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size):
    setup(rank, world_size)
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
        solarize=0.0,
        saturation=0.0,
        contrast=0.0,
        brightness=0.0,
        sharpness=0.0,
        hue=0.0,
        posterization=4,
        rotation=10,
        erasing=0.0,
        affine=0.0,
        crop=0.0,
        gaussian_noise=0.0,
    )
    # Transforms
    transforms_creator = TransformsCreator(parameter_storage)
    transforms_creator.create_transforms()
    # Load the dataset
    dataset_loader = DatasetLoader(parameter_storage, transforms=transforms_creator)
    # Create the distributed data loaders
    data_loader_creator = DataLoaderCreator(parameter_storage, dataset_loader)
    data_loader_creator.create_dist_dataloaders(rank, world_size)
    # Create the model
    device = "cuda"
    efficient_net = CustomEfficientNet(type="b4", num_classes=7)
    efficient_net.to(rank)
    efficient_net = DistributedDataParallel(
        efficient_net, device_ids=[rank], output_device=rank, find_unused_parameters=True
    )
    criterion = nn.CrossEntropyLoss()
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
        efficient_net, metrics={"acc": Accuracy(), "loss": Loss(nn.CrossEntropyLoss())}, device=device
    )
    evaluator.run(data_loader_creator.validation_dataloader)
    print(evaluator.state.metrics)
    ax = lr_finder.plot()
    ax.figure.savefig("lr_finder_plot.png")
    print("Suggestion", lr_finder.lr_suggestion())
    cleanup()


if __name__ == "__main__":
    # Number of gpus
    world_size = 2
    mp.spawn(main, args=(world_size,), nprocs=world_size)
