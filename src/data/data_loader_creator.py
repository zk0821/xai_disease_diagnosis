from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import torch


class DataLoaderCreator:

    def __init__(self, parameter_storage, dataset_loader):
        self.parameter_storage = parameter_storage
        self.dataset_loader = dataset_loader
        self.train_dataloader = None
        self.validation_dataloader = None
        self.test_dataloader = None

    def create_dataloaders(self):
        # Dataloaders
        if self.parameter_storage.do_oversampling:
            class_sample_count = np.array(
                [
                    len(np.where(self.dataset_loader.train_dataframe["type"] == t)[0])
                    for t in np.unique(self.dataset_loader.train_dataframe["type"])
                ]
            )
            weight = 1.0 / class_sample_count
            samples_weight = np.array([weight[t] for t in self.dataset_loader.train_dataframe["type"]])
            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            weighted_random_sampler = WeightedRandomSampler(
                weights=samples_weight,
                num_samples=len(samples_weight),
                replacement=True,
            )
            self.train_dataloader = DataLoader(
                self.dataset_loader.train_dataset,
                batch_size=self.parameter_storage.batch_size,
                shuffle=False,
                num_workers=8,
                sampler=weighted_random_sampler,
            )
        else:
            self.train_dataloader = DataLoader(
                self.dataset_loader.train_dataset,
                batch_size=self.parameter_storage.batch_size,
                shuffle=True,
                num_workers=16,
            )
        self.validation_dataloader = DataLoader(
            self.dataset_loader.validation_dataset,
            batch_size=self.parameter_storage.batch_size,
            shuffle=False,
            num_workers=16,
        )
        self.test_dataloader = DataLoader(
            self.dataset_loader.test_dataset,
            batch_size=self.parameter_storage.batch_size,
            shuffle=False,
            num_workers=16,
        )

    def create_dataloaders_from_ids(self, train_ids, validation_ids):
        train_subsampler = SubsetRandomSampler(train_ids)
        self.train_dataloader = DataLoader(
            self.dataset_loader.full_train_dataset,
            batch_size=self.parameter_storage.batch_size,
            sampler=train_subsampler,
            num_workers=16,
        )
        validation_subsampler = SubsetRandomSampler(validation_ids)
        self.validation_dataloader = DataLoader(
            self.dataset_loader.full_train_dataset,
            batch_size=self.parameter_storage.batch_size,
            sampler=validation_subsampler,
            num_workers=16,
        )
        self.test_dataloader = DataLoader(
            self.dataset_loader.test_dataset,
            batch_size=self.parameter_storage.batch_size,
            shuffle=False,
            num_workers=16,
        )

    def create_dist_dataloaders(self, rank, world_size):
        # Distributed Sampler
        sampler = DistributedSampler(
            self.dataset_loader.train_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        # Dataloader
        self.train_dataloader = DataLoader(
            self.dataset_loader.train_dataset,
            batch_size=self.parameter_storage.batch_size,
            pin_memory=False,
            num_workers=0,
            drop_last=False,
            shuffle=False,
            sampler=sampler,
        )
