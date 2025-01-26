from torchvision.transforms import v2
import torch


class TransformsCreator:

    def __init__(self, parameter_storage):
        self.parameter_storage = parameter_storage
        self.train_transforms = None
        self.validation_transforms = None
        self.test_transforms = None

    def create_transforms(self):

        mean = [0.5325749, 0.37914315, 0.39562908]
        spread = [0.36953005, 0.2814345, 0.29810122]
        self.train_transforms = v2.Compose(
            [
                v2.Resize(self.parameter_storage.size),
                v2.RandomHorizontalFlip(0.2),
                v2.RandomVerticalFlip(0.2),
                v2.RandomRotation(90),
                v2.RandomAffine(10),
                v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                v2.Normalize(mean, spread),
            ]
        )
        self.validation_transforms = v2.Compose(
            [
                v2.Resize(self.parameter_storage.size),
                v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                v2.Normalize(mean, spread),
            ]
        )
        self.test_transforms = v2.Compose(
            [
                v2.Resize(self.parameter_storage.size),
                v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                v2.Normalize(mean, spread),
            ]
        )
