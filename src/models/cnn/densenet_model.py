import torch
import torch.nn as nn

from models.template_model import TemplateModel


class CustomDenseNet(TemplateModel):
    def __init__(self, type, num_classes):
        super().__init__(type, num_classes)

    def load_pretrained(self):
        if self.type == "densenet121":
            return torch.hub.load("pytorch/vision", "densenet121", weights="IMAGENET1K_V1")
        elif self.type == "densenet161":
            return torch.hub.load("pytorch/vision", "densenet161", weights="IMAGENET1K_V1")
        elif self.type == "densenet169":
            return torch.hub.load("pytorch/vision", "densenet169", weights="IMAGENET1K_V1")
        elif self.type == "densenet201":
            return torch.hub.load("pytorch/vision", "densenet201", weights="IMAGENET1K_V1")
        else:
            raise RuntimeError(f'Type "{self.type}" is not supported for model DenseNet!')

    def remove_final_layer(self):
        self.pretrained_model.classifier = nn.Sequential()

    def get_num_features_of_pretrained_model(self):
        return self.pretrained_model.classifier.in_features

    def forward(self, x):
        return super().forward(x)

    def unfreeze_pretrained_layers(self):
        super().unfreeze_pretrained_layers()
