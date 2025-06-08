from data.ham10000_dataset import HAM10000Dataframe, HAM10000Dataset
from sklearn.model_selection import train_test_split


class DatasetLoader:

    def __init__(self, parameter_storage, train_transforms=None, validation_transforms=None, test_transforms=None):
        if parameter_storage.dataset == "HAM_10000":
            print("Selected dataset: ISIC 2018 (HAM10000)")
            print(f"Selected policy: {parameter_storage.augmentation_policy}")

            self.full_train_dataframe = HAM10000Dataframe(path="data/ham10000/train", csv_name="groundtruth.csv")
            self.full_train_dataset = HAM10000Dataset(
                path=self.full_train_dataframe.path,
                dataframe=self.full_train_dataframe.get_dataframe(),
                transforms=train_transforms,
                policy=parameter_storage.augmentation_policy,
            )

            self.classes = self.full_train_dataframe.categories.categories.tolist()
            self.num_classes = self.full_train_dataframe.categories.categories.size

            if True:
                # Validation dataset is created from the train dataset
                self.train_dataframe, self.validation_dataframe = train_test_split(
                    self.full_train_dataframe.get_dataframe(),
                    test_size=parameter_storage.validation_split,
                    random_state=42,
                    stratify=self.full_train_dataframe.get_dataframe()["type"],
                )
                self.train_dataset = HAM10000Dataset(
                    path=self.full_train_dataframe.path,
                    dataframe=self.train_dataframe,
                    transforms=train_transforms,
                    policy=parameter_storage.augmentation_policy,
                )
                self.validation_dataset = HAM10000Dataset(
                    path=self.full_train_dataframe.path,
                    dataframe=self.validation_dataframe,
                    transforms=validation_transforms,
                    policy="multi_crop",
                )
            else:
                # Validation dataset is provided, but is too small to be used
                self.train_dataframe = self.full_train_dataframe
                self.train_dataset = self.full_train_dataset
                self.validation_dataframe = HAM10000Dataframe(
                    path="data/ham10000/validation", csv_name="groundtruth.csv"
                )
                self.validation_dataset = HAM10000Dataset(
                    path=self.validation_dataframe.path,
                    dataframe=self.validation_dataframe.get_dataframe(),
                    transforms=transforms.validation_transforms,
                )

            self.test_dataframe = HAM10000Dataframe(path="data/ham10000/test", csv_name="groundtruth.csv")
            self.test_dataset = HAM10000Dataset(
                path=self.test_dataframe.path,
                dataframe=self.test_dataframe.get_dataframe(),
                transforms=test_transforms,
                policy="multi_crop",
            )
        else:
            raise RuntimeError(f"Unsupported dataset: {parameter_storage.dataset}")
