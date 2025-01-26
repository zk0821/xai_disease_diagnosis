import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import json
import wandb

from utils.early_stoppage import EarlyStoppage

from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Trainer:
    def __init__(self, parameter_storage, model, dataset_loader, data_loader_creator, evaluator) -> None:
        self.parameter_storage = parameter_storage
        self.model = model
        self.dataset_loader = dataset_loader
        self.data_loader_creator = data_loader_creator
        self.evaluator = evaluator
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.early_stoppage = None
        self.epochs = self.parameter_storage.epochs
        self.device = torch.device("cuda")

    # def create_dataloaders(self):
    #     # Dataloaders
    #     if self.do_oversampling:
    #         class_sample_count = np.array(
    #             [
    #                 len(np.where(self.train_df["type"] == t)[0])
    #                 for t in np.unique(self.train_df["type"])
    #             ]
    #         )
    #         weight = 1.0 / class_sample_count
    #         samples_weight = np.array([weight[t] for t in self.train_df["type"]])
    #         samples_weight = torch.from_numpy(samples_weight)
    #         samples_weight = samples_weight.double()
    #         self.weighted_random_sampler = WeightedRandomSampler(
    #             weights=samples_weight,
    #             num_samples=len(samples_weight),
    #             replacement=True,
    #         )
    #     mean = [0.76303804, 0.54694057, 0.57165635]
    #     spread = [0.14156434, 0.15333284, 0.17053322]
    #     train_transforms = v2.Compose(
    #         [
    #             v2.Resize(self.size),
    #             v2.RandomHorizontalFlip(),
    #             v2.RandomVerticalFlip(),
    #             v2.RandomRotation(20),
    #             v2.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
    #             v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    #             v2.Normalize(mean, spread),
    #         ]
    #     )
    #     train_dataset = HAM10000Dataset(
    #         self.train_df, img_dir="data/ham10000/images", transform=train_transforms
    #     )
    #     if self.do_oversampling:
    #         self.train_dataloader = DataLoader(
    #             train_dataset,
    #             batch_size=32,
    #             shuffle=False,
    #             num_workers=8,
    #             sampler=self.weighted_random_sampler,
    #         )
    #     else:
    #         self.train_dataloader = DataLoader(
    #             train_dataset, batch_size=32, shuffle=True, num_workers=8
    #         )

    #     validation_transforms = v2.Compose(
    #         [
    #             v2.Resize(self.size),
    #             v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    #             v2.Normalize(mean, spread),
    #         ]
    #     )
    #     validation_dataset = HAM10000Dataset(
    #         self.validation_df,
    #         img_dir="data/ham10000/images",
    #         transform=validation_transforms,
    #     )
    #     self.validation_dataloader = DataLoader(
    #         validation_dataset, batch_size=32, shuffle=False, num_workers=8
    #     )

    #     test_transforms = v2.Compose(
    #         [
    #             v2.Resize(self.size),
    #             v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    #             v2.Normalize(mean, spread),
    #         ]
    #     )
    #     test_dataset = HAM10000Dataset(
    #         self.test_df, img_dir="data/ham10000/images", transform=test_transforms
    #     )
    #     self.test_dataloader = DataLoader(
    #         test_dataset, batch_size=32, shuffle=False, num_workers=8
    #     )

    def prepare_model(self):
        self.model.unfreeze_pretrained_layers()
        print("Using: ", torch.cuda.device_count(), "GPUs!")
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        if self.parameter_storage.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.parameter_storage.learning_rate,
                weight_decay=self.parameter_storage.weight_decay,
            )
        elif self.parameter_storage.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.parameter_storage.learning_rate,
                weight_decay=self.parameter_storage.weight_decay,
            )
        else:
            raise RuntimeError(
                f"Unsupported optimizer: {self.parameter_storage.optimizer}"
            )

        if self.parameter_storage.do_class_weights:
            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.unique(self.dataset_loader.train_dataframe["type"].to_numpy()),
                y=self.dataset_loader.train_dataframe["type"].to_numpy(),
            )
            class_weights = torch.tensor(class_weights, dtype=torch.float)
            print("Class Weights")
            print(class_weights)
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights
            ).to(self.device)
        else:
            if self.parameter_storage.criterion == "cross_entropy":
                self.criterion = nn.CrossEntropyLoss().to(self.device)
            else:
                raise RuntimeError(
                    f"Unsupported criterion: {self.parameter_storage.criterion}"
                )

        if self.parameter_storage.scheduler == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=20, threshold=1e-4
            )
        elif self.parameter_storage.scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )
        elif self.parameter_storage.scheduler == "none":
            self.scheduler = None
        else:
            raise RuntimeError(
                f"Unsupported scheduler: {self.parameter_storage.scheduler}"
            )
        self.early_stoppage = EarlyStoppage(patience=10, min_delta=0.1)

    def load_model(self):
        self.model.load_state_dict(
            torch.load(
                f"models/{self.parameter_storage.model_architecture}/{self.parameter_storage.model_type}/{self.parameter_storage.name}.pth"
            )
        )

    def train_model(self):
        for epoch in range(1, self.epochs + 1):
            # Reset train evaluator at start of epoch
            self.evaluator.train_evaluator.reset()
            with tqdm(self.data_loader_creator.train_dataloader, unit="batch") as prog:
                prog.set_description(f"Epoch {epoch}/{self.epochs}")
                self.model.train()
                for _, data in enumerate(self.data_loader_creator.train_dataloader):
                    prog.update()
                    # Batch data
                    images, labels = data
                    self.evaluator.train_evaluator.record_labels(labels)
                    images = Variable(images).to(self.device)
                    labels = Variable(labels).to(self.device)

                    # Get outputs
                    self.optimizer.zero_grad()
                    outputs = self.model(images)

                    # Calculate loss and perform backpropagation
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    # Make prediction
                    _, predictions = outputs.max(dim=1)
                    self.evaluator.train_evaluator.record_predictions(predictions.detach().cpu())

                    # Calculate probabilities
                    probabilities = torch.softmax(outputs, dim=1)
                    self.evaluator.train_evaluator.record_probabilities(probabilities.detach().cpu())

                    # Calculate batch accuracy and loss
                    batch_accuracy = predictions.eq(labels).sum().item() / predictions.size(dim=0)
                    batch_loss = loss.item()
                    self.evaluator.train_evaluator.record_loss(batch_loss)

                    prog.set_postfix(
                        {
                            "Train Batch Accuracy": batch_accuracy,
                            "Train Batch Loss": batch_loss,
                        }
                    )
                # Validation
                self.evaluate_model(
                    self.data_loader_creator.validation_dataloader,
                    self.evaluator.validation_evaluator
                )
                prog.set_postfix(
                    {
                        "Train Micro Accuracy": self.evaluator.train_evaluator.micro_accuracy(),
                        "Train Macro Accuracy": self.evaluator.train_evaluator.macro_accuracy(),
                        "Train Balanced Accuracy": self.evaluator.train_evaluator.balanced_accuracy(),
                        "Train Loss": self.evaluator.train_evaluator.loss(),
                        "Train Macro AUC": self.evaluator.train_evaluator.macro_auc(),
                        "Train Macro AP": self.evaluator.train_evaluator.macro_average_precision(),
                        "Validation Micro Accuracy": self.evaluator.validation_evaluator.micro_accuracy(),
                        "Validation Macro Accuracy": self.evaluator.validation_evaluator.macro_accuracy(),
                        "Validation Balanced Accuracy": self.evaluator.validation_evaluator.balanced_accuracy(),
                        "Validation Loss": self.evaluator.validation_evaluator.loss(),
                        "Validation Macro AUC": self.evaluator.validation_evaluator.macro_auc(),
                        "Validation Macro AP": self.evaluator.validation_evaluator.macro_average_precision()
                    }
                )
                prog.refresh()
                wandb.log(
                    {
                        "train/micro_accuracy": self.evaluator.train_evaluator.micro_accuracy(),
                        "train/macro_accuracy": self.evaluator.train_evaluator.macro_accuracy(),
                        "train/balanced_accuracy": self.evaluator.train_evaluator.balanced_accuracy(),
                        "train/loss": self.evaluator.train_evaluator.loss(),
                        "train/macro_auc": self.evaluator.train_evaluator.macro_auc(),
                        "train/macro_ap": self.evaluator.train_evaluator.macro_average_precision(),
                        "validation/micro_accuracy": self.evaluator.validation_evaluator.micro_accuracy(),
                        "validation/macro_accuracy": self.evaluator.validation_evaluator.macro_accuracy(),
                        "validation/balanced_accuracy": self.evaluator.validation_evaluator.balanced_accuracy(),
                        "validation/loss": self.evaluator.validation_evaluator.loss(),
                        "validation/macro_auc": self.evaluator.validation_evaluator.macro_auc(),
                        "validation/macro_ap": self.evaluator.validation_evaluator.macro_average_precision(),
                        "epoch": epoch,
                    }
                )
                # Scheduler
                if self.scheduler is not None:
                    self.scheduler.step(self.evaluator.validation_evaluator.loss())
                # Model Checkpoint
                if self.evaluator.validation_evaluator.loss() < self.early_stoppage.get_min_validation_loss():
                    print(
                        f"Best validation loss detected at epoch {epoch}! Validation Loss: {self.evaluator.validation_evaluator.loss()}. Saving model.."
                    )
                    torch.save(
                        self.model.state_dict(),
                        f"models/{self.parameter_storage.model_architecture}/{self.parameter_storage.model_type}/{self.parameter_storage.name}.pth",
                    )
                # Early Stoppage
                if self.early_stoppage.early_stop(self.evaluator.validation_evaluator.loss()):
                    print("Early stoppage...")
                    break


    def evaluate_model(self, dataloader, evaluator):
        self.model.eval()
        # Reset evaluator at start
        evaluator.reset()
        with torch.no_grad():
            for _, data in enumerate(dataloader):
                # Batch data
                images, labels = data
                evaluator.record_labels(labels)
                images = Variable(images).to(self.device)
                labels = Variable(labels).to(self.device)

                outputs = self.model(images)
                _, predictions = outputs.max(dim=1)
                evaluator.record_predictions(predictions.detach().cpu())

                probabilities = torch.softmax(outputs, dim=1)
                evaluator.record_probabilities(probabilities.detach().cpu())

                batch_loss = self.criterion(outputs, labels).item()
                evaluator.record_loss(batch_loss)
                # ROC AUC
            #     if ix == 0:
            #         all_labels = labels.cpu().numpy()
            #         all_predictions = predictions.cpu().numpy()
            #         all_probs = torch.softmax(outputs, dim=1).cpu().numpy()
            #     else:
            #         all_labels = np.concatenate(
            #             (all_labels, labels.cpu().numpy()), axis=0
            #         )
            #         all_predictions = np.concatenate(
            #             (all_predictions, predictions.cpu().numpy()), axis=0
            #         )
            #         all_probs = np.concatenate(
            #             (all_probs, torch.softmax(outputs, dim=1).cpu().numpy()), axis=0
            #         )
            # auc_macro_ovr = roc_auc_score(
            #     all_labels, all_probs, average="macro", multi_class="ovr"
            # )
            # auc_macro_ovo = roc_auc_score(
            #     all_labels, all_probs, average="macro", multi_class="ovo"
            # )
            # ap_macro = average_precision_score(all_labels, all_probs, average="macro")
            # print(
            #     f"AUC Macro OVR: {auc_macro_ovr}, \
            #     AUC Macro OVO: {auc_macro_ovo}, \
            #     AP Macro: {ap_macro}"
            # )
            # wandb.log(
            #     {
            #         "AUC Macro OVR": auc_macro_ovr,
            #         "AUC Macro OVO": auc_macro_ovo,
            #         "AP Macro": ap_macro,
            #     }
            # )
            # Build confusion matrix
            # cf_matrix = confusion_matrix(all_labels, all_predictions)
            # classes = self.ham_df_object.get_categories().categories
            # new_classes = []
            # for clss in classes:
            #     new_class = clss.replace(" ", "\n")
            #     new_classes.append(new_class)
            # df_cm = pd.DataFrame(
            #     cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
            #     index=[i for i in new_classes],
            #     columns=[i for i in new_classes],
            # )
            # plt.figure(figsize=(15, 10))
            # sns.heatmap(df_cm, annot=True)
            # plt.savefig(f"models/{self.model_type}/confusion_matrix.png")

    def test_model(self):
        self.evaluate_model(
            self.data_loader_creator.test_dataloader,
            self.evaluator.test_evaluator
        )
        wandb.log(
            {
                "test/micro_accuracy": self.evaluator.test_evaluator.micro_accuracy(),
                "test/macro_accuracy": self.evaluator.test_evaluator.macro_accuracy(),
                "test/balanced_accuracy": self.evaluator.test_evaluator.balanced_accuracy(),
                "test/loss": self.evaluator.test_evaluator.loss(),
                "test/macro_auc": self.evaluator.test_evaluator.macro_auc(),
                "test/macro_ap": self.evaluator.test_evaluator.macro_average_precision()
            }
        )
        # Per class accuracies
        per_class_accuracies = self.evaluator.test_evaluator.per_class_accuracies()
        column_arr = ["Class", "Accuracy"]
        data_arr = []
        for c in per_class_accuracies:
            data_arr.append((c, per_class_accuracies[c]))
        wandb.log(
            {
                "test/per_class_accuracy": wandb.Table(data=data_arr, columns=column_arr)
            }
        )
        # Per class AUCs
        per_class_aucs = self.evaluator.test_evaluator.per_class_auc()
        column_arr = ["Class", "AUC"]
        data_arr = []
        for c in per_class_aucs:
            data_arr.append((c, per_class_aucs[c]))
        wandb.log(
            {
                "test/per_class_auc": wandb.Table(data=data_arr, columns=column_arr)
            }
        )
        # Per class APs
        per_class_aps = self.evaluator.test_evaluator.per_class_average_precision()
        column_arr = ["Class", "AP"]
        data_arr = []
        for c in per_class_aps:
            data_arr.append((c, per_class_aps[c]))
        wandb.log(
            {
                "test/per_class_ap": wandb.Table(data=data_arr, columns=column_arr)
            }
        )
        # Classification report
        report_columns = ["Class", "Precision", "Recall", "F1-Score", "Support"]
        report_table = []
        report = self.evaluator.test_evaluator.classification_report()
        report = report.split('\n')
        for line in report[2:(7 + 2)]:
            print(line.split())
            report_table.append(line.split())
        for line in report[10:13]:
            cols = line.split()
            if len(cols) == 6:
                label = cols[0]
                cols.pop(0)
                cols.pop(0)
                if label == "macro":
                    cols.insert(0, "macro_avg")
                elif label == "weighted":
                    cols.insert(0, "weighted_avg")
                elif label == "micro":
                    cols.insert(0, "micro_avg")
            if len(cols) == 3:
                cols.insert(1, "/")
                cols.insert(1, "/")
            report_table.append(cols)
        wandb.log(
            {
                "test/classification_report": wandb.Table(data=report_table, columns=report_columns)
            }
        )

    def dump_metrics(self):
        # Save training data
        with open(f"models/{self.model_type}/train_acc.txt", "w") as train_acc_file:
            json.dump(self.total_acc_train, train_acc_file)
        with open(f"models/{self.model_type}/train_loss.txt", "w") as train_loss_file:
            json.dump(self.total_loss_train, train_loss_file)
        with open(f"models/{self.model_type}/val_acc.txt", "w") as val_acc_file:
            json.dump(self.total_acc_val, val_acc_file)
        with open(f"models/{self.model_type}/val_loss.txt", "w") as val_loss_file:
            json.dump(self.total_loss_val, val_loss_file)
