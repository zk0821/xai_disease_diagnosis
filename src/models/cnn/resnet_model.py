import torch
import torch.nn as nn

from models.template_model import TemplateModel

class CustomResNet(TemplateModel):
    def __init__(self, type, num_classes):
        super().__init__(type, num_classes)

    def load_pretrained(self):
        if self.type == "resnet50":
            return torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
        elif self.type == "resnext50":
            return torch.hub.load("pytorch/vision", "resnext50_32x4d", weights="IMAGENET1K_V2")
        elif self.type == "resnet101":
            return torch.hub.load(
                "pytorch/vision", "resnet101", weights="IMAGENET1K_V2"
            )
        elif self.type == "resnet152":
            return torch.hub.load(
                "pytorch/vision", "resnet152", weights="IMAGENET1K_V2"
            )
        else:
            raise RuntimeError('Type "{self.type}" is not supported for model ResNet!')

    def remove_final_layer(self):
        self.pretrained_model.fc = nn.Sequential()

    def get_num_features_of_pretrained_model(self):
        return self.pretrained_model.fc.in_features

    def forward(self, x):
        return super().forward(x)

    def unfreeze_pretrained_layers(self):
        super().unfreeze_pretrained_layers()
