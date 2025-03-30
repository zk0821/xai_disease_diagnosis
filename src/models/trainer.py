import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import v2

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import json
import wandb
import random
import csv

from utils.early_stoppage import EarlyStoppage


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
        self.device = torch.device("cuda:0")

    def prepare_model(self):
        self.model.unfreeze_pretrained_layers()
        print("Using: ", torch.cuda.device_count(), "GPU(s)!")
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
            raise RuntimeError(f"Unsupported optimizer: {self.parameter_storage.optimizer}")

        if self.parameter_storage.do_class_weights:
            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.unique(self.dataset_loader.train_dataframe["type"].to_numpy()),
                y=self.dataset_loader.train_dataframe["type"].to_numpy(),
            )
            class_weights = torch.tensor(class_weights, dtype=torch.float)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.0).to(self.device)
        else:
            if self.parameter_storage.criterion == "cross_entropy":
                self.criterion = nn.CrossEntropyLoss(label_smoothing=0.0).to(self.device)
            else:
                raise RuntimeError(f"Unsupported criterion: {self.parameter_storage.criterion}")

        if self.parameter_storage.scheduler == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.1, patience=3, threshold=1e-4
            )
        elif self.parameter_storage.scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        elif self.parameter_storage.scheduler == "none":
            self.scheduler = None
        else:
            raise RuntimeError(f"Unsupported scheduler: {self.parameter_storage.scheduler}")
        self.early_stoppage = EarlyStoppage(patience=7, min_delta=0.1, max_patience=11)

    def load_model(self):
        self.model.load_state_dict(
            torch.load(
                f"models/{self.parameter_storage.model_architecture}/{self.parameter_storage.model_type}/{self.parameter_storage.name}.pth"
            )
        )

    def train_model(self, log_wandb=True):
        assert self.optimizer is not None
        assert self.criterion is not None
        assert self.early_stoppage is not None
        for epoch in range(1, self.epochs + 1):
            # Reset train evaluator at start of epoch
            self.evaluator.train_evaluator.reset()
            with tqdm(self.data_loader_creator.train_dataloader, unit="batch") as prog:
                prog.set_description(f"Epoch {epoch}/{self.epochs}")
                self.model.train()
                for _, data in enumerate(self.data_loader_creator.train_dataloader):
                    prog.update()
                    # Batch data
                    images, labels, paths = data
                    self.evaluator.train_evaluator.record_labels(labels)

                    # MixUp -> linearly add images, but don't change the labels
                    # if random.uniform(0.0, 1.0) <= 0.3:
                    #    mixup = v2.MixUp(alpha=random.uniform(0.0, 0.1), num_classes=self.evaluator.num_classes)
                    #    images, _ = mixup(images, labels)

                    images = images.to(self.device)
                    labels = labels.to(self.device)

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
                self.evaluate_model(self.data_loader_creator.validation_dataloader, self.evaluator.validation_evaluator)
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
                        "Validation Macro AP": self.evaluator.validation_evaluator.macro_average_precision(),
                    }
                )
                prog.refresh()
                if log_wandb:
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

    def evaluate_model(self, dataloader, evaluator, with_augmentation=False):
        assert self.criterion is not None
        print("Model evaluation with augmentation:", with_augmentation)
        self.model.eval()
        # num_augmentations = 16
        # Reset evaluator at start
        evaluator.reset()
        # prediction_csv = open("test_predictions.csv", "w")
        with torch.no_grad():
            for _, data in enumerate(dataloader):
                # Batch data
                images, labels, _ = data
                evaluator.record_labels(labels)
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predictions = outputs.max(dim=1)
                evaluator.record_predictions(predictions.detach().cpu())

                probabilities = torch.softmax(outputs, dim=1)
                evaluator.record_probabilities(probabilities.detach().cpu())

                batch_loss = self.criterion(outputs, labels).item()
                evaluator.record_loss(batch_loss)
            # Test Time Augmentation
            # csv_writer = csv.writer(prediction_csv)
            # csv_writer.writerow(["image", "AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"])
            # final_predictions = None
            # for i in range(num_augmentations if with_augmentation else 1):
            #    temporary_predictions = None
            #    for _, data in enumerate(dataloader):
            # Batch data
            #       images, labels, paths = data
            #        if i == 0:
            #            evaluator.record_labels(labels)
            #        images = images.to(self.device)
            #        labels = labels.to(self.device)

            #        outputs = self.model(images)
            #        for o in outputs.detach().cpu():
            #            if temporary_predictions is None:
            #                temporary_predictions = o
            #            else:
            #                temporary_predictions = np.vstack((temporary_predictions, o))
            #    if final_predictions is None:
            #        final_predictions = temporary_predictions
            #    else:
            #        final_predictions += temporary_predictions
            # final_predictions /= num_augmentations if with_augmentation else 1
            # final_outputs = final_predictions
            # softmax_probs = nn.functional.softmax(final_outputs, dim=1)
            # csv_writer.writerow(
            #    [
            #        paths[0],
            #        str(softmax_probs[0][0].item()),
            #        str(softmax_probs[0][1].item()),
            #        str(softmax_probs[0][2].item()),
            #        str(softmax_probs[0][3].item()),
            #        str(softmax_probs[0][4].item()),
            #        str(softmax_probs[0][5].item()),
            #        str(softmax_probs[0][6].item()),
            #    ]
            # )
            # _, predictions = final_outputs.max(dim=1)
            # predictions = final_outputs.argmax(axis=1)
            # evaluator.record_predictions(torch.from_numpy(predictions))

            # probabilities = torch.softmax(final_outputs, dim=1)
            # evaluator.record_probabilities(probabilities)

            # batch_loss = self.criterion(final_outputs, my_labels).item()
            # evaluator.record_loss(batch_loss)
        # prediction_csv.close()

    def test_model(self, with_augmentation=False, log_wandb=True):
        # First test on the validation set
        self.evaluate_model(
            self.data_loader_creator.validation_dataloader, self.evaluator.validation_evaluator, with_augmentation
        )
        if log_wandb:
            wandb.log(
                {
                    "best_validation/micro_accuracy": self.evaluator.validation_evaluator.micro_accuracy(),
                    "best_validation/macro_accuracy": self.evaluator.validation_evaluator.macro_accuracy(),
                    "best_validation/balanced_accuracy": self.evaluator.validation_evaluator.balanced_accuracy(),
                    # "best_validation/loss": self.evaluator.validation_evaluator.loss(),
                    # "best_validation/macro_auc": self.evaluator.validation_evaluator.macro_auc(),
                    # "best_validation/macro_ap": self.evaluator.validation_evaluator.macro_average_precision(),
                }
            )
        else:
            print(f"Best validation BMCA: {self.evaluator.validation_evaluator.balanced_accuracy()}")
        self.evaluate_model(self.data_loader_creator.test_dataloader, self.evaluator.test_evaluator, with_augmentation)
        if log_wandb:
            wandb.log(
                {
                    "test/micro_accuracy": self.evaluator.test_evaluator.micro_accuracy(),
                    "test/macro_accuracy": self.evaluator.test_evaluator.macro_accuracy(),
                    "test/balanced_accuracy": self.evaluator.test_evaluator.balanced_accuracy(),
                    # "test/loss": self.evaluator.test_evaluator.loss(),
                    # "test/macro_auc": self.evaluator.test_evaluator.macro_auc(),
                    # "test/macro_ap": self.evaluator.test_evaluator.macro_average_precision(),
                }
            )
        else:
            print(f"Test BMCA: {self.evaluator.test_evaluator.balanced_accuracy()}")
        return
        # Per class accuracies
        per_class_accuracies = self.evaluator.test_evaluator.per_class_accuracies()
        column_arr = ["Class", "Accuracy"]
        data_arr = []
        for c in per_class_accuracies:
            data_arr.append((c, per_class_accuracies[c]))
        if log_wandb:
            wandb.log({"test/per_class_accuracy": wandb.Table(data=data_arr, columns=column_arr)})
        # Per class AUCs
        per_class_aucs = self.evaluator.test_evaluator.per_class_auc()
        column_arr = ["Class", "AUC"]
        data_arr = []
        for c in per_class_aucs:
            data_arr.append((c, per_class_aucs[c]))
        if log_wandb:
            wandb.log({"test/per_class_auc": wandb.Table(data=data_arr, columns=column_arr)})
        # Per class APs
        per_class_aps = self.evaluator.test_evaluator.per_class_average_precision()
        column_arr = ["Class", "AP"]
        data_arr = []
        for c in per_class_aps:
            data_arr.append((c, per_class_aps[c]))
        if log_wandb:
            wandb.log({"test/per_class_ap": wandb.Table(data=data_arr, columns=column_arr)})
        # Classification report
        report_columns = ["Class", "Precision", "Recall", "F1-Score", "Support"]
        report_table = []
        report = self.evaluator.test_evaluator.classification_report()
        report = report.split("\n")
        for line in report[2 : (7 + 2)]:
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
        if log_wandb:
            wandb.log({"test/classification_report": wandb.Table(data=report_table, columns=report_columns)})

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
