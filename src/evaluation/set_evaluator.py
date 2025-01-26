import torch
import numpy as np

from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score, average_precision_score


class SetEvaluator:

    def __init__(self, classes, num_classes):
        self.classes = classes
        self.num_classes = num_classes
        self.predictions = torch.tensor([])
        self.probabilities = torch.tensor([])
        self.labels = torch.tensor([])
        self.cumulative_loss = 0
        self.num_batches = 0

    def record_probabilities(self, probabilities):
        self.probabilities = torch.cat((self.probabilities, probabilities))

    def record_predictions(self, predictions):
        self.predictions = torch.cat((self.predictions, predictions))

    def record_labels(self, labels):
        self.labels = torch.cat((self.labels, labels))

    def record_loss(self, loss):
        self.cumulative_loss += loss
        self.num_batches += 1

    def reset(self):
        self.predictions = torch.tensor([])
        self.probabilities = torch.tensor([])
        self.labels = torch.tensor([])
        self.cumulative_loss = 0
        self.num_batches = 0

    def loss(self):
        final_loss = self.cumulative_loss / self.num_batches
        return final_loss

    def micro_accuracy(self):
        num_true_positives = self.predictions.eq(self.labels).sum().item()
        micro_accuracy = num_true_positives / self.predictions.size(dim=0)
        return micro_accuracy

    def per_class_accuracies(self):
        # Get the confusion matrix
        cm = confusion_matrix(self.labels, self.predictions)

        # We will store the results in a dictionary for easy access later
        per_class_accuracies = {}

        # Calculate the accuracy for each one of our classes
        for idx in range(self.num_classes):
            # True negatives are all the samples that are not our current GT class (not the current row)
            # and were not predicted as the current class (not the current column)
            true_negatives = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))

            # True positives are all the samples of our current GT class that were predicted as such
            true_positives = cm[idx, idx]

            # The accuracy for the current class is the ratio between correct predictions to all predictions
            per_class_accuracies[self.classes[idx]] = (true_positives + true_negatives) / np.sum(cm)

        return per_class_accuracies

    def macro_accuracy(self):
        per_class_acc = self.per_class_accuracies()
        macro_acc = 0
        for c in per_class_acc:
            macro_acc += per_class_acc[c]
        macro_acc /= self.num_classes
        return macro_acc

    def balanced_accuracy(self):
        balanced_accuracy = balanced_accuracy_score(self.labels, self.predictions)
        return balanced_accuracy

    def classification_report(self):
        report = classification_report(y_true=self.labels, y_pred=self.predictions, target_names=self.classes)
        print(report)
        return report

    def per_class_auc(self):
        per_class_auc = {}
        for idx in range(self.num_classes):
            other_class = [x for x in range(self.num_classes) if x != idx]
            new_labels = [0 if x in other_class else 1 for x in self.labels]
            new_probs = self.probabilities[:, idx]
            roc_auc = roc_auc_score(new_labels, new_probs)
            per_class_auc[self.classes[idx]] = roc_auc
        return per_class_auc

    def macro_auc(self):
        per_class_aucs = self.per_class_auc()
        macro_auc = 0
        for c in per_class_aucs:
            macro_auc += per_class_aucs[c]
        macro_auc /= self.num_classes
        return macro_auc

    def roc(self):
        print("ROC")

    def per_class_average_precision(self):
        per_class_aps = {}
        for idx in range(self.num_classes):
            other_class = [x for x in range(self.num_classes) if x != idx]
            new_labels = [0 if x in other_class else 1 for x in self.labels]
            new_probs = self.probabilities[:, idx]
            ap = average_precision_score(new_labels, new_probs)
            per_class_aps[self.classes[idx]] = ap
        return per_class_aps

    def macro_average_precision(self):
        per_class_aps = self.per_class_average_precision()
        macro_ap = 0
        for c in per_class_aps:
            macro_ap += per_class_aps[c]
        macro_ap /= self.num_classes
        return macro_ap
