from data.aptos_dataset import AptosDataframe, AptosDataset
from data.chest_dataset import ChestDataframe, ChestDataset
from data.isic_dataset import ISICDataframe, ISICDataset
from sklearn.model_selection import StratifiedKFold


class DatasetLoader:

    def __init__(self, parameter_storage, train_transforms=None, validation_transforms=None, test_transforms=None):
        print(
            f"Selected train policy: {parameter_storage.train_augmentation_policy}; test policy: {parameter_storage.test_augmentation_policy}"
        )
        create_validation_from_training = True
        if parameter_storage.dataset == "HAM_10000":
            print("Selected dataset: ISIC 2018 (HAM10000)")
            self.full_train_dataframe = ISICDataframe(path="data/ham10000/train", csv_name="groundtruth.csv")
            self.full_train_dataset = ISICDataset(
                path=self.full_train_dataframe.path,
                dataframe=self.full_train_dataframe.get_dataframe(),
                policy=parameter_storage.train_augmentation_policy,
            )

            self.classes = self.full_train_dataframe.categories.categories.tolist()
            self.num_classes = self.full_train_dataframe.categories.categories.size

            if create_validation_from_training:
                print("Using validation dataset created from training data")
                # Validation dataset is created from the train dataset using KFold
                kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=parameter_storage.random_seed)
                (train_ids, validation_ids) = next(
                    kfold.split(
                        self.full_train_dataframe.get_dataframe(), self.full_train_dataframe.get_dataframe()["type"]
                    )
                )
                self.train_dataframe = self.full_train_dataframe.get_dataframe().loc[train_ids]
                self.train_dataset = ISICDataset(
                    path=self.full_train_dataframe.path,
                    dataframe=self.train_dataframe,
                    policy=parameter_storage.train_augmentation_policy,
                )
                self.validation_dataframe = self.full_train_dataframe.get_dataframe().loc[validation_ids]
                self.validation_dataset = ISICDataset(
                    path=self.full_train_dataframe.path,
                    dataframe=self.validation_dataframe,
                    policy=parameter_storage.test_augmentation_policy,
                )
            else:
                print("Using official validation set")
                # Validation dataset is provided, but is too small to be used
                self.train_dataframe = self.full_train_dataframe.get_dataframe()
                self.train_dataset = self.full_train_dataset
                self.validation_dataframe = ISICDataframe(path="data/ham10000/validation", csv_name="groundtruth.csv")
                self.validation_dataset = ISICDataset(
                    path=self.validation_dataframe.path,
                    dataframe=self.validation_dataframe.get_dataframe(),
                    policy=parameter_storage.test_augmentation_policy,
                )

            self.test_dataframe = ISICDataframe(path="data/ham10000/test", csv_name="groundtruth.csv")
            self.test_dataset = ISICDataset(
                path=self.test_dataframe.path,
                dataframe=self.test_dataframe.get_dataframe(),
                policy=parameter_storage.test_augmentation_policy,
            )
        elif parameter_storage.dataset == "ISIC_2017":
            print("Selected dataset: ISIC 2017")
            self.full_train_dataframe = ISICDataframe(path="data/isic2017/train", csv_name="groundtruth.csv")
            self.full_train_dataset = ISICDataset(
                path=self.full_train_dataframe.path,
                dataframe=self.full_train_dataframe.get_dataframe(),
                policy=parameter_storage.train_augmentation_policy,
            )

            self.classes = self.full_train_dataframe.categories.categories.tolist()
            self.num_classes = self.full_train_dataframe.categories.categories.size

            if create_validation_from_training:
                print("Using validation dataset created from training data")
                # Validation dataset is created from the train dataset using KFold
                kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=parameter_storage.random_seed)
                (train_ids, validation_ids) = next(
                    kfold.split(
                        self.full_train_dataframe.get_dataframe(), self.full_train_dataframe.get_dataframe()["type"]
                    )
                )
                self.train_dataframe = self.full_train_dataframe.get_dataframe().loc[train_ids]
                self.train_dataset = ISICDataset(
                    path=self.full_train_dataframe.path,
                    dataframe=self.train_dataframe,
                    policy=parameter_storage.train_augmentation_policy,
                )
                self.validation_dataframe = self.full_train_dataframe.get_dataframe().loc[validation_ids]
                self.validation_dataset = ISICDataset(
                    path=self.full_train_dataframe.path,
                    dataframe=self.validation_dataframe,
                    policy=parameter_storage.test_augmentation_policy,
                )
            else:
                raise RuntimeError("Official validation set not supported!")

            self.test_dataframe = ISICDataframe(path="data/isic2017/test", csv_name="groundtruth.csv")
            self.test_dataset = ISICDataset(
                path=self.test_dataframe.path,
                dataframe=self.test_dataframe.get_dataframe(),
                policy=parameter_storage.test_augmentation_policy,
            )
        elif parameter_storage.dataset == "ISIC_2019":
            print("Selected dataset: ISIC 2019")
            self.full_train_dataframe = ISICDataframe(path="data/isic2019/train", csv_name="groundtruth.csv")
            self.full_train_dataset = ISICDataset(
                path=self.full_train_dataframe.path,
                dataframe=self.full_train_dataframe.get_dataframe(),
                policy=parameter_storage.train_augmentation_policy,
            )

            self.classes = self.full_train_dataframe.categories.categories.tolist()
            self.num_classes = self.full_train_dataframe.categories.categories.size

            if create_validation_from_training:
                print("Using validation dataset created from training data")
                # Validation dataset is created from the train dataset using KFold
                kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=parameter_storage.random_seed)
                (train_ids, validation_ids) = next(
                    kfold.split(
                        self.full_train_dataframe.get_dataframe(), self.full_train_dataframe.get_dataframe()["type"]
                    )
                )
                self.train_dataframe = self.full_train_dataframe.get_dataframe().loc[train_ids]
                self.train_dataset = ISICDataset(
                    path=self.full_train_dataframe.path,
                    dataframe=self.train_dataframe,
                    policy=parameter_storage.train_augmentation_policy,
                )
                self.validation_dataframe = self.full_train_dataframe.get_dataframe().loc[validation_ids]
                self.validation_dataset = ISICDataset(
                    path=self.full_train_dataframe.path,
                    dataframe=self.validation_dataframe,
                    policy=parameter_storage.test_augmentation_policy,
                )
            else:
                raise RuntimeError("Official validation set not supported!")

            self.test_dataframe = ISICDataframe(path="data/isic2019/test", csv_name="groundtruth.csv")
            self.test_dataset = ISICDataset(
                path=self.test_dataframe.path,
                dataframe=self.test_dataframe.get_dataframe(),
                policy=parameter_storage.test_augmentation_policy,
            )
        elif parameter_storage.dataset == "APTOS2019":
            print("Selected dataset: APTOS2019")
            self.full_train_dataframe = AptosDataframe(path="data/aptos2019/train", csv_name="groundtruth.csv")
            self.full_train_dataset = AptosDataset(
                path=self.full_train_dataframe.path,
                dataframe=self.full_train_dataframe.get_dataframe(),
                policy=parameter_storage.train_augmentation_policy,
            )

            # No DR, Mild, Moderate, Severe, Proliferative DR
            self.classes = [0, 1, 2, 3, 4]
            self.num_classes = 5

            if create_validation_from_training:
                print("Using validation dataset created from training data")
                # Validation dataset is created from the train dataset using KFold
                kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=parameter_storage.random_seed)
                (train_ids, validation_ids) = next(
                    kfold.split(
                        self.full_train_dataframe.get_dataframe(), self.full_train_dataframe.get_dataframe()["type"]
                    )
                )
                self.train_dataframe = self.full_train_dataframe.get_dataframe().loc[train_ids]
                self.train_dataset = AptosDataset(
                    path=self.full_train_dataframe.path,
                    dataframe=self.train_dataframe,
                    policy=parameter_storage.train_augmentation_policy,
                )
                self.validation_dataframe = self.full_train_dataframe.get_dataframe().loc[validation_ids]
                self.validation_dataset = AptosDataset(
                    path=self.full_train_dataframe.path,
                    dataframe=self.validation_dataframe,
                    policy=parameter_storage.test_augmentation_policy,
                )
            else:
                raise RuntimeError("Official validation set not supported!")

            self.test_dataframe = AptosDataframe(path="data/aptos2019/test", csv_name="groundtruth.csv")
            self.test_dataset = AptosDataset(
                path=self.test_dataframe.path,
                dataframe=self.test_dataframe.get_dataframe(),
                policy=parameter_storage.test_augmentation_policy,
            )
        elif parameter_storage.dataset == "ChestXRAY":
            print("Selected dataset: Chest Pneumonia XRAY")
            self.full_train_dataframe = ChestDataframe(path="data/chest_xray/train", csv_name="groundtruth.csv")
            self.full_train_dataset = ChestDataset(
                path=self.full_train_dataframe.path,
                dataframe=self.full_train_dataframe.get_dataframe(),
                policy=parameter_storage.train_augmentation_policy,
            )

            self.classes = [0, 1]
            self.num_classes = 2

            if create_validation_from_training:
                print("Using validation dataset created from training data")
                # Validation dataset is created from the train dataset using KFold
                kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=parameter_storage.random_seed)
                (train_ids, validation_ids) = next(
                    kfold.split(
                        self.full_train_dataframe.get_dataframe(), self.full_train_dataframe.get_dataframe()["type"]
                    )
                )
                self.train_dataframe = self.full_train_dataframe.get_dataframe().loc[train_ids]
                self.train_dataset = ChestDataset(
                    path=self.full_train_dataframe.path,
                    dataframe=self.train_dataframe,
                    policy=parameter_storage.train_augmentation_policy,
                )
                self.validation_dataframe = self.full_train_dataframe.get_dataframe().loc[validation_ids]
                self.validation_dataset = ChestDataset(
                    path=self.full_train_dataframe.path,
                    dataframe=self.validation_dataframe,
                    policy=parameter_storage.test_augmentation_policy,
                )
            else:
                print("Using official validation set")
                # Validation dataset is provided, but is too small to be used
                self.train_dataframe = self.full_train_dataframe.get_dataframe()
                self.train_dataset = self.full_train_dataset
                self.validation_dataframe = ChestDataframe(
                    path="data/chest_xray/validation", csv_name="groundtruth.csv"
                )
                self.validation_dataset = ChestDataset(
                    path=self.validation_dataframe.path,
                    dataframe=self.validation_dataframe.get_dataframe(),
                    policy=parameter_storage.test_augmentation_policy,
                )

            self.test_dataframe = ChestDataframe(path="data/chest_xray/test", csv_name="groundtruth.csv")
            self.test_dataset = ChestDataset(
                path=self.test_dataframe.path,
                dataframe=self.test_dataframe.get_dataframe(),
                policy=parameter_storage.test_augmentation_policy,
            )

        else:
            raise RuntimeError(f"Unsupported dataset: {parameter_storage.dataset}")
