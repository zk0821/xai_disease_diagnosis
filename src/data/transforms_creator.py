from torchvision.transforms import v2
import torch
import random


class TransformsCreator:

    def __init__(self, parameter_storage, tta=False):
        self.parameter_storage = parameter_storage
        self.train_transforms = None
        self.validation_transforms = None
        self.test_transforms = None
        self.tta = tta

    def create_transforms(self):
        # Low-Cost Augment Paper: https://arxiv.org/abs/2101.02353
        mean = [0.5325749, 0.37914315, 0.39562908]
        spread = [0.36953005, 0.2814345, 0.29810122]
        if False:
            self.train_transforms = v2.Compose(
                [
                    v2.RandomSolarize(
                        self.parameter_storage.solarize, self.parameter_storage.augmentation_probability
                    ),  # Solarize
                    v2.RandomApply(
                        [v2.ColorJitter(saturation=self.parameter_storage.saturation)],
                        p=self.parameter_storage.augmentation_probability,
                    ),  # Color
                    v2.RandomApply(
                        [v2.ColorJitter(contrast=self.parameter_storage.contrast)],
                        p=self.parameter_storage.augmentation_probability,
                    ),  # Contrast
                    v2.RandomApply(
                        [v2.ColorJitter(brightness=self.parameter_storage.brightness)],
                        p=self.parameter_storage.augmentation_probability,
                    ),  # Brightness
                    v2.RandomAdjustSharpness(
                        self.parameter_storage.sharpness, p=self.parameter_storage.augmentation_probability
                    ),  # Sharpness
                    # v2.RandomApply([v2.ColorJitter(hue=self.parameter_storage.hue)], p=0.3),  # Color shift
                    v2.RandomEqualize(p=self.parameter_storage.augmentation_probability),  # Equalize
                    v2.RandomPosterize(
                        bits=self.parameter_storage.posterization, p=self.parameter_storage.augmentation_probability
                    ),  # Posterize
                    v2.RandomAutocontrast(p=self.parameter_storage.augmentation_probability),  # Autocontrast
                    v2.RandomApply(
                        [v2.RandomRotation(degrees=self.parameter_storage.rotation)],
                        p=self.parameter_storage.augmentation_probability,
                    ),  # Rotate
                    v2.RandomHorizontalFlip(p=self.parameter_storage.augmentation_probability),  # Horizontal flip
                    v2.RandomVerticalFlip(p=self.parameter_storage.augmentation_probability),  # Vertical flip
                    v2.RandomErasing(
                        p=self.parameter_storage.augmentation_probability, scale=(0, self.parameter_storage.erasing)
                    ),  # Cutout
                    v2.RandomApply(
                        [
                            v2.RandomAffine(
                                degrees=0,
                                shear=(
                                    -self.parameter_storage.affine,
                                    self.parameter_storage.affine,
                                    -self.parameter_storage.affine,
                                    self.parameter_storage.affine,
                                ),
                            )
                        ],
                        p=self.parameter_storage.augmentation_probability,
                    ),  # Shear X + Shear Y
                    v2.RandomResizedCrop(self.parameter_storage.size, scale=self.parameter_storage.crop),  # Scale
                    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                    # v2.RandomApply(
                    #    [v2.GaussianNoise(mean=0.0, sigma=self.parameter_storage.gaussian_noise, clip=True)], 0.3
                    # ),  # Gaussian Noise
                    v2.Normalize(mean, spread),
                ]
            )
        elif False:
            self.train_transforms = v2.Compose(
                [
                    v2.Resize(self.parameter_storage.size),
                    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                    v2.Normalize(mean, spread),
                ]
            )
        else:
            # Test RandAugment
            self.train_transforms = v2.Compose(
                [
                    v2.RandAugment(num_ops=2, magnitude=5),
                    v2.RandomResizedCrop(self.parameter_storage.size),
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
