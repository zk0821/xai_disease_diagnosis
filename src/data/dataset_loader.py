from data.ham10000_dataset import HAM10000Dataframe, HAM10000Dataset
from sklearn.model_selection import train_test_split


class DatasetLoader:

    def __init__(self, parameter_storage, transforms):
        if parameter_storage.dataset == "HAM_10000":
            self.full_train_dataframe = HAM10000Dataframe(path="data/ham10000/train", csv_name="groundtruth.csv")
            self.full_train_dataset = HAM10000Dataset(
                path=self.full_train_dataframe.path,
                dataframe=self.full_train_dataframe.get_dataframe(),
                transforms=transforms.train_transforms,
            )
            self.classes = self.full_train_dataframe.categories.categories.tolist()
            self.num_classes = self.full_train_dataframe.categories.categories.size

            if True:
                self.train_dataframe, self.validation_dataframe = train_test_split(
                    self.full_train_dataframe.get_dataframe(),
                    test_size=0.2,
                    random_state=42,
                    stratify=self.full_train_dataframe.get_dataframe()["type"],
                )
                self.train_dataset = HAM10000Dataset(
                    path=self.full_train_dataframe.path,
                    dataframe=self.train_dataframe,
                    transforms=transforms.train_transforms,
                )
                self.validation_dataset = HAM10000Dataset(
                    path=self.full_train_dataframe.path,
                    dataframe=self.validation_dataframe,
                    transforms=transforms.validation_transforms,
                )
            else:
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
                transforms=transforms.test_transforms,
            )
        else:
            assert False
