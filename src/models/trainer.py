import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import v2

import cv2
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import json
import wandb
import random
import csv

from utils.early_stoppage import EarlyStoppage
from utils.model_checkpoint import ModelCheckpoint
from models.loss.custom_loss import FocalLoss, ClassBalancedFocalLoss, ClassBalancedCrossEntropy, LDAMLoss

DRW_EPOCH=20


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
        # class weights
        self.cls_num_list = None
        self.per_cls_weights = None
        self.use_weights = False

    def define_optimizer(self):
        print(f"Chosen 'optimizer' option: {self.parameter_storage.optimizer}")
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
            raise RuntimeError(f"Unsupported 'optimizer' option: {self.parameter_storage.optimizer}")

    def define_class_weights(self, curr_epoch):
        # Class Weights
        print(f"Chosen 'class_weights' option: {self.parameter_storage.class_weights}")
        cls_num_list = np.array(
            [
                len(np.where(self.dataset_loader.train_dataframe["type"] == t)[0])
                for t in np.unique(self.dataset_loader.train_dataframe["type"])
            ]
        )
        if self.parameter_storage.class_weights == "none":
            per_cls_weights = None
            self.use_weights = False
        elif self.parameter_storage.class_weights == "reweight":
            self.use_weights = True
            # Weights per class
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)
        elif self.parameter_storage.class_weights == "drw":
            # Weights per class
            if curr_epoch < DRW_EPOCH:
                self.use_weights = False
                per_cls_weights = None
            else:
                self.use_weights = True
                beta = 0.9999
                effective_num = 1.0 - np.power(beta, cls_num_list)
                per_cls_weights = (1.0 - beta) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)
        else:
            raise RuntimeError(f"Unsupported 'class_weights' option: {self.parameter_storage.class_weights}")
        self.cls_num_list = cls_num_list
        self.per_cls_weights = per_cls_weights

    def define_criterion(self):
       # Criterion
        print(f"Chosen 'criterion' option: {self.parameter_storage.criterion}")
        if self.parameter_storage.criterion == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss(weight=self.per_cls_weights).to(self.device)
        elif self.parameter_storage.criterion == "cb_cross_entropy":
            self.criterion = ClassBalancedCrossEntropy(
                beta=self.parameter_storage.class_balance_beta, samples_per_class=self.cls_num_list, use_weights=self.use_weights
            )
        elif self.parameter_storage.criterion == "focal_loss":
            self.criterion = FocalLoss(
                alpha=self.per_cls_weights, gamma=self.parameter_storage.focal_loss_gamma, reduction="mean"
            )
        elif self.parameter_storage.criterion == "cb_focal_loss":
            self.criterion = ClassBalancedFocalLoss(
                beta=self.parameter_storage.class_balance_beta,
                gamma=self.parameter_storage.focal_loss_gamma,
                samples_per_class=self.cls_num_list,
                use_weights=self.use_weights
            )
        elif self.parameter_storage.criterion == "ldam":
            self.criterion = LDAMLoss(cls_num_list=self.cls_num_list, max_m=0.5, s=30, weight=self.per_cls_weights).to(
                self.device
            )
        else:
            raise RuntimeError(f"Unsupported 'criterion' option: {self.parameter_storage.criterion}")

    def define_scheduler(self):
        # Scheduler
        print(f"Chosen 'scheduler' option: {self.parameter_storage.scheduler}")
        if self.parameter_storage.scheduler == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.1, patience=8, threshold=1e-4
            )
        elif self.parameter_storage.scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif self.parameter_storage.scheduler == "multi_step":
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 40, 50], gamma=0.1)
        elif self.parameter_storage.scheduler == "none":
            self.scheduler = None
        else:
            raise RuntimeError(f"Unsupported scheduler: {self.parameter_storage.scheduler}")

    def prepare_model(self):
        # Prepare model
        self.model.unfreeze_pretrained_layers()
        print("Using: ", torch.cuda.device_count(), "GPU(s)!")
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        # Prepare learning strategy
        self.define_optimizer()
        self.define_class_weights(0)
        self.define_criterion()
        self.define_scheduler()
        self.model_checkpoint = ModelCheckpoint(enabled=self.parameter_storage.model_checkpoint, use_loss=False)
        self.early_stoppage = EarlyStoppage(enabled=self.parameter_storage.early_stoppage, patience=12, min_delta=0.1, max_patience=12)

    def load_model(self):
        self.model.load_state_dict(
            torch.load(
                f"models/{self.parameter_storage.model_architecture}/{self.parameter_storage.model_type}/{self.parameter_storage.name}.pth"
            )
        )

    def train_model(self, log_wandb=True, fold=None):
        assert self.optimizer is not None
        assert self.criterion is not None
        assert self.early_stoppage is not None
        for epoch in range(1, self.epochs + 1):
            # Reset train evaluator at start of epoch
            self.evaluator.train_evaluator.reset()
            # If Deferred-Reweighting re-calculate weights and re-define criterion
            if self.parameter_storage.class_weights == "drw":
                self.define_class_weights(epoch)
                self.define_criterion()
            with tqdm(self.data_loader_creator.train_dataloader, unit="batch") as prog:
                prog.set_description(f"Epoch {epoch}/{self.epochs}")
                self.model.train()
                for _, data in enumerate(self.data_loader_creator.train_dataloader):
                    prog.update()
                    # Batch data
                    images, labels, paths = data
                    self.evaluator.train_evaluator.record_labels(labels)

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
                    if fold is not None:
                        wandb.log(
                            {
                                f"{fold}/train/micro_accuracy": self.evaluator.train_evaluator.micro_accuracy(),
                                f"{fold}/train/macro_accuracy": self.evaluator.train_evaluator.macro_accuracy(),
                                f"{fold}/train/balanced_accuracy": self.evaluator.train_evaluator.balanced_accuracy(),
                                f"{fold}/train/loss": self.evaluator.train_evaluator.loss(),
                                f"{fold}/train/macro_auc": self.evaluator.train_evaluator.macro_auc(),
                                f"{fold}/train/macro_ap": self.evaluator.train_evaluator.macro_average_precision(),
                                f"{fold}/validation/micro_accuracy": self.evaluator.validation_evaluator.micro_accuracy(),
                                f"{fold}/validation/macro_accuracy": self.evaluator.validation_evaluator.macro_accuracy(),
                                f"{fold}/validation/balanced_accuracy": self.evaluator.validation_evaluator.balanced_accuracy(),
                                f"{fold}/validation/loss": self.evaluator.validation_evaluator.loss(),
                                f"{fold}/validation/macro_auc": self.evaluator.validation_evaluator.macro_auc(),
                                f"{fold}/validation/macro_ap": self.evaluator.validation_evaluator.macro_average_precision(),
                                f"{fold}/epoch": epoch,
                            }
                        )
                    else:
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
                    if self.parameter_storage.scheduler == "plateau":
                        self.scheduler.step(self.evaluator.validation_evaluator.loss())
                    else:
                        self.scheduler.step()
                # Model Checkpoint
                if self.model_checkpoint.save_checkpoint(self.evaluator.validation_evaluator.loss(), self.evaluator.validation_evaluator.balanced_accuracy()):
                    print(
                        f"[Epoch {epoch}] Validation Loss={self.evaluator.validation_evaluator.loss()}, Validation Balanced Accuracy={self.evaluator.validation_evaluator.balanced_accuracy()}. Saving model.."
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
        assert self.criterion is not None
        self.model.eval()
        # Reset evaluator at start
        evaluator.reset()
        with torch.no_grad():
            for _, data in enumerate(dataloader):
                # Batch data
                images, labels, _ = data
                evaluator.record_labels(labels)

                if self.parameter_storage.test_augmentation_policy == "multi_crop":
                    all_crops = []
                    num_crops = 16
                    for image in images:
                        crops = []
                        for index in range(num_crops):
                            C, H, W = image.shape
                            l_region = 1.0
                            s_region = 0.8
                            y_n = index // 4
                            x_n = index % 4

                            # Determine crop region size
                            if W >= H:
                                x_region = int(W * l_region)
                                y_region = int(H * s_region)
                            else:
                                x_region = int(W * s_region)
                                y_region = int(H * l_region)

                            x_region = max(x_region, 224)
                            y_region = max(y_region, 224)

                            # Center the region
                            x_cut = (W - x_region) // 2
                            y_cut = (H - y_region) // 2

                            # Compute crop top-left coordinates
                            x_loc = x_cut + int(x_n * (x_region - 224) / (4 - 1))
                            y_loc = y_cut + int(y_n * (y_region - 224) / (4 - 1))

                            # Apply crop
                            crop = image[:, y_loc : y_loc + 224, x_loc : x_loc + 224]
                            crops.append(crop)
                        crops = torch.stack(crops)
                        all_crops.append(crops)
                    all_crops_batch = torch.cat(all_crops, dim=0)
                    all_crops_batch = all_crops_batch.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(all_crops_batch)
                    outputs = outputs.view(images.shape[0], num_crops, 7)
                    outputs = outputs.mean(dim=1)
                else:
                    labels = labels.to(self.device)
                    images = images.to(self.device)
                    outputs = self.model(images)

                _, predictions = outputs.max(dim=1)
                evaluator.record_predictions(predictions.detach().cpu())

                probabilities = torch.softmax(outputs, dim=1)
                evaluator.record_probabilities(probabilities.detach().cpu())

                batch_loss = self.criterion(outputs, labels).item()
                evaluator.record_loss(batch_loss)

    def test_model(self, log_wandb=True, fold=None):
        # First test on the validation set
        self.evaluate_model(
            self.data_loader_creator.validation_dataloader, self.evaluator.validation_evaluator,
        )
        if log_wandb:
            if fold is not None:
                wandb.log(
                    {
                        f"{fold}/best_validation/micro_accuracy": self.evaluator.validation_evaluator.micro_accuracy(),
                        f"{fold}/best_validation/macro_accuracy": self.evaluator.validation_evaluator.macro_accuracy(),
                        f"{fold}/best_validation/balanced_accuracy": self.evaluator.validation_evaluator.balanced_accuracy(),
                        f"{fold}/best_validation/loss": self.evaluator.validation_evaluator.loss(),
                        f"{fold}/best_validation/macro_auc": self.evaluator.validation_evaluator.macro_auc(),
                        f"{fold}/best_validation/macro_ap": self.evaluator.validation_evaluator.macro_average_precision(),
                    }
                )
            else:
                wandb.log(
                    {
                        "best_validation/micro_accuracy": self.evaluator.validation_evaluator.micro_accuracy(),
                        "best_validation/macro_accuracy": self.evaluator.validation_evaluator.macro_accuracy(),
                        "best_validation/balanced_accuracy": self.evaluator.validation_evaluator.balanced_accuracy(),
                        "best_validation/loss": self.evaluator.validation_evaluator.loss(),
                        "best_validation/macro_auc": self.evaluator.validation_evaluator.macro_auc(),
                        "best_validation/macro_ap": self.evaluator.validation_evaluator.macro_average_precision(),
                    }
                )
        else:
            print(f"Best validation BMCA: {self.evaluator.validation_evaluator.balanced_accuracy()}")
        # Secondly test on the test set
        self.evaluate_model(self.data_loader_creator.test_dataloader, self.evaluator.test_evaluator)
        if log_wandb:
            if fold is not None:
                wandb.log(
                    {
                        f"{fold}/test/micro_accuracy": self.evaluator.test_evaluator.micro_accuracy(),
                        f"{fold}/test/macro_accuracy": self.evaluator.test_evaluator.macro_accuracy(),
                        f"{fold}/test/balanced_accuracy": self.evaluator.test_evaluator.balanced_accuracy(),
                        f"{fold}/test/loss": self.evaluator.test_evaluator.loss(),
                        f"{fold}/test/macro_auc": self.evaluator.test_evaluator.macro_auc(),
                        f"{fold}/test/macro_ap": self.evaluator.test_evaluator.macro_average_precision(),
                    }
                )
            else:
                wandb.log(
                    {
                        "test/micro_accuracy": self.evaluator.test_evaluator.micro_accuracy(),
                        "test/macro_accuracy": self.evaluator.test_evaluator.macro_accuracy(),
                        "test/balanced_accuracy": self.evaluator.test_evaluator.balanced_accuracy(),
                        "test/loss": self.evaluator.test_evaluator.loss(),
                        "test/macro_auc": self.evaluator.test_evaluator.macro_auc(),
                        "test/macro_ap": self.evaluator.test_evaluator.macro_average_precision(),
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
            if fold is not None:
                wandb.log({f"{fold}/test/per_class_accuracy": wandb.Table(data=data_arr, columns=column_arr)})
            else:
                wandb.log({"test/per_class_accuracy": wandb.Table(data=data_arr, columns=column_arr)})
        # Per class AUCs
        per_class_aucs = self.evaluator.test_evaluator.per_class_auc()
        column_arr = ["Class", "AUC"]
        data_arr = []
        for c in per_class_aucs:
            data_arr.append((c, per_class_aucs[c]))
        if log_wandb:
            if fold is not None:
                wandb.log({f"{fold}/test/per_class_auc": wandb.Table(data=data_arr, columns=column_arr)})
            else:
                wandb.log({"test/per_class_auc": wandb.Table(data=data_arr, columns=column_arr)})
        # Per class APs
        per_class_aps = self.evaluator.test_evaluator.per_class_average_precision()
        column_arr = ["Class", "AP"]
        data_arr = []
        for c in per_class_aps:
            data_arr.append((c, per_class_aps[c]))
        if log_wandb:
            if fold is not None:
                wandb.log({f"{fold}/test/per_class_ap": wandb.Table(data=data_arr, columns=column_arr)})
            else:
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
            if fold is not None:
                wandb.log({f"{fold}/test/classification_report": wandb.Table(data=report_table, columns=report_columns)})
            else:
                wandb.log({"test/classification_report": wandb.Table(data=report_table, columns=report_columns)})