import torch
import torch.nn as nn

from models.template_model import TemplateModel

class CustomConvNext(TemplateModel):
    def __init__(self, type, num_classes):
        super().__init__(type, num_classes)

    def load_pretrained(self):
        if self.type == "convnext_small":
            return torch.hub.load(
                "pytorch/vision", "convnext_small", weights="IMAGENET1K_V1"
            )
        elif self.type == "convnext_base":
            return torch.hub.load(
                "pytorch/vision", "convnext_base", weights="IMAGENET1K_V1"
            )
        elif self.type == "convnext_large":
            return torch.hub.load(
                "pytorch/vision", "convnext_large", weights="IMAGENET1K_V1"
            )
        else:
            raise RuntimeError(
                'Type "{self.type}" is not supported for model ConvNext!'
            )

    def remove_final_layer(self):
        self.pretrained_model.classifier[2] = nn.Sequential()

    def get_num_features_of_pretrained_model(self):
        return self.pretrained_model.classifier[2].in_features

    def forward(self, x):
        return super().forward(x)

    def unfreeze_pretrained_layers(self):
        super().unfreeze_pretrained_layers()
