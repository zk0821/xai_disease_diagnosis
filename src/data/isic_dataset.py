import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
import pandas as pd
import numpy as np
import torch

from data.augmentation_policies import apply_policy
from data.augmentation_functions import RandomCropInRate


class ISICDataset(Dataset):
    def __init__(self, path, dataframe, policy=None) -> None:
        self.path = path
        self.dataframe = dataframe
        self.policy = policy

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Read the image
        img_path = f"{self.path}/images/{self.dataframe['image'].iloc[idx]}.jpg"
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Make sure image is 650x400
        image = cv2.resize(image, (650, 400))
        image = torch.from_numpy(np.array(image, dtype=np.uint8))
        # Read the label
        label = torch.tensor(int(self.dataframe["type"].iloc[idx]))
        # Apply policy
        if self.policy is not None:
            image_for_mixup = (
                f"{self.path}/images/{self.dataframe['image'].iloc[np.random.randint(0, len(self.dataframe))]}.jpg"
            )
            mixup_image = cv2.imread(image_for_mixup)
            mixup_image = cv2.cvtColor(mixup_image, cv2.COLOR_BGR2RGB)
            mixup_image = torch.from_numpy(np.array(mixup_image, dtype=np.uint8))
            image = apply_policy(self.policy, image, mixup_image)
            if self.policy != "resize" and self.policy != "multi_crop":
                # Perform Random Crop with size 224x224
                image = Image.fromarray(image.numpy())
                crop_method = RandomCropInRate(nsize=(224, 224), rand_rate=(0.8, 1.0))
                image = crop_method(image)
            else:
                image = Image.fromarray(image.numpy())
            # Transform image to tensor
            transformation = transforms.Compose([transforms.ToTensor()])
            image = transformation(image)
        return image, label, img_path


class ISICDataframe:
    def __init__(self, path, csv_name):
        self.path = path
        # Read the csv file
        self.metadata_df = pd.read_csv(os.path.join(self.path, csv_name))
        if self.metadata_df.shape[1] > 10:
            # ISIC 2019 Test has unneeded columns
            self.metadata_df = self.metadata_df.iloc[:, :-2]
        # Get classes
        class_names = sorted(list(self.metadata_df.columns[1:]))
        # Add categorical type
        self.metadata_df["diagnosis"] = self.metadata_df.iloc[:, 1:].idxmax(axis=1)
        self.categories = pd.Categorical(self.metadata_df["diagnosis"], categories=class_names)
        print(self.categories.categories)
        self.metadata_df["type"] = self.categories.codes
        # Assertions
        if "image" not in self.metadata_df.columns:
            self.metadata_df["image"] = self.metadata_df["image_id"]
        assert self.metadata_df["image"].duplicated().any() is not False

    def get_dataframe(self):
        return self.metadata_df

    def get_categories(self):
        return self.categories

    def print_diagnosis_counts(self):
        grouped_df = self.metadata_df.groupby("type")
        print(grouped_df["type"].count())
