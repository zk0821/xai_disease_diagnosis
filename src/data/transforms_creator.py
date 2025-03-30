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
        # Note: MixUp done during training phase as a batch is required!
        if False:
            self.train_transforms = v2.Compose(
                [
                    v2.RandomSolarize(self.parameter_storage.solarize, 0.3),  # Solarize
                    v2.RandomApply([v2.ColorJitter(saturation=self.parameter_storage.saturation)], p=0.3),  # Color
                    v2.RandomApply([v2.ColorJitter(contrast=self.parameter_storage.contrast)], p=0.3),  # Contrast
                    v2.RandomApply([v2.ColorJitter(brightness=self.parameter_storage.brightness)], p=0.3),  # Brightness
                    v2.RandomAdjustSharpness(self.parameter_storage.sharpness, p=0.3),  # Sharpness
                    # v2.RandomApply([v2.ColorJitter(hue=self.parameter_storage.hue)], p=0.3),  # Color shift
                    v2.RandomEqualize(p=0.3),  # Equalize
                    v2.RandomPosterize(bits=self.parameter_storage.posterization, p=0.3),  # Posterize
                    v2.RandomAutocontrast(p=0.3),  # Autocontrast
                    v2.RandomApply([v2.RandomRotation(degrees=self.parameter_storage.rotation)], p=0.3),  # Rotate
                    v2.RandomHorizontalFlip(p=0.3),  # Horizontal flip
                    v2.RandomVerticalFlip(p=0.3),  # Vertical flip
                    v2.RandomErasing(p=0.3, scale=(0, self.parameter_storage.erasing)),  # Cutout
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
                        p=0.3,
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
                    v2.Resize(self.parameter_storage.size),
                    v2.RandAugment(num_ops=5, magnitude=14),
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
        if self.tta:
            self.test_transforms = v2.Compose(
                [
                    v2.RandomSolarize(self.parameter_storage.solarize, 0.3),  # Solarize
                    v2.RandomApply([v2.ColorJitter(saturation=self.parameter_storage.saturation)], p=0.3),  # Color
                    v2.RandomApply([v2.ColorJitter(contrast=self.parameter_storage.contrast)], p=0.3),  # Contrast
                    v2.RandomApply([v2.ColorJitter(brightness=self.parameter_storage.brightness)], p=0.3),  # Brightness
                    v2.RandomAdjustSharpness(self.parameter_storage.sharpness, p=0.3),  # Sharpness
                    # v2.RandomApply([v2.ColorJitter(hue=self.parameter_storage.hue)], p=0.3),  # Color shift
                    v2.RandomEqualize(p=0.3),  # Equalize
                    v2.RandomPosterize(bits=self.parameter_storage.posterization, p=0.3),  # Posterize
                    v2.RandomAutocontrast(p=0.3),  # Autocontrast
                    v2.RandomApply([v2.RandomRotation(degrees=self.parameter_storage.rotation)], p=0.3),  # Rotate
                    v2.RandomHorizontalFlip(p=0.3),  # Horizontal flip
                    v2.RandomVerticalFlip(p=0.3),  # Vertical flip
                    v2.RandomErasing(p=0.3, scale=(0, self.parameter_storage.erasing)),  # Cutout
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
                        p=0.3,
                    ),  # Shear X + Shear Y
                    v2.RandomResizedCrop(self.parameter_storage.size, scale=self.parameter_storage.crop),  # Scale
                    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                    # v2.RandomApply(
                    #    [v2.GaussianNoise(mean=0.0, sigma=self.parameter_storage.gaussian_noise, clip=True)], 0.3
                    # ),  # Gaussian Noise
                    v2.Normalize(mean, spread),
                ]
            )
        else:
            self.test_transforms = v2.Compose(
                [
                    v2.Resize(self.parameter_storage.size),
                    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                    v2.Normalize(mean, spread),
                ]
            )
