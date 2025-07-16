from data.ham10000_dataset import HAM10000Dataframe, HAM10000Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold


class DatasetLoader:

    def __init__(self, parameter_storage, train_transforms=None, validation_transforms=None, test_transforms=None):
        if parameter_storage.dataset == "HAM_10000":
            print("Selected dataset: ISIC 2018 (HAM10000)")
            print(f"Selected train policy: {parameter_storage.train_augmentation_policy}; test policy: {parameter_storage.test_augmentation_policy}")

            self.full_train_dataframe = HAM10000Dataframe(path="data/ham10000/train", csv_name="groundtruth.csv")
            self.full_train_dataset = HAM10000Dataset(
                path=self.full_train_dataframe.path,
                dataframe=self.full_train_dataframe.get_dataframe(),
                policy=parameter_storage.train_augmentation_policy,
            )

            self.classes = self.full_train_dataframe.categories.categories.tolist()
            self.num_classes = self.full_train_dataframe.categories.categories.size

            create_validation_from_training = True

            if create_validation_from_training:
                print("Using validation dataset created from training data")
                # Validation dataset is created from the train dataset using KFold
                kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=parameter_storage.random_seed)
                (train_ids, validation_ids) = next(kfold.split(
                        self.full_train_dataframe.get_dataframe(),
                        self.full_train_dataframe.get_dataframe()["type"]
                ))
                self.train_dataframe = self.full_train_dataframe.get_dataframe().loc[train_ids]
                print("train dataframe:", self.train_dataframe)
                self.train_dataset = HAM10000Dataset(
                    path=self.full_train_dataframe.path,
                    dataframe=self.train_dataframe,
                    policy=parameter_storage.train_augmentation_policy,
                )
                self.validation_dataframe = self.full_train_dataframe.get_dataframe().loc[validation_ids]
                print("validation dataframe:", self.validation_dataframe)
                self.validation_dataset = HAM10000Dataset(
                    path=self.full_train_dataframe.path,
                    dataframe=self.validation_dataframe,
                    policy=parameter_storage.test_augmentation_policy
                )
            else:
                print("Using official validation set")
                # Validation dataset is provided, but is too small to be used
                self.train_dataframe = self.full_train_dataframe.get_dataframe()
                self.train_dataset = self.full_train_dataset
                self.validation_dataframe = HAM10000Dataframe(
                    path="data/ham10000/validation", csv_name="groundtruth.csv"
                )
                self.validation_dataset = HAM10000Dataset(
                    path=self.validation_dataframe.path,
                    dataframe=self.validation_dataframe.get_dataframe(),
                    policy=parameter_storage.test_augmentation_policy
                )

            self.test_dataframe = HAM10000Dataframe(path="data/ham10000/test", csv_name="groundtruth.csv")
            self.test_dataset = HAM10000Dataset(
                path=self.test_dataframe.path,
                dataframe=self.test_dataframe.get_dataframe(),
                policy=parameter_storage.test_augmentation_policy
            )
        else:
            raise RuntimeError(f"Unsupported dataset: {parameter_storage.dataset}")
