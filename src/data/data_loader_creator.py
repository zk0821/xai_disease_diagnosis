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
            print("Oversampling selected: WeightedRandomSampler")
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
                num_workers=16,
                sampler=weighted_random_sampler,
            )
        else:
            print("Oversampling selected: None")
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
