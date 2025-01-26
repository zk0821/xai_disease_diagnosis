import torch
import torch.nn as nn

from models.template_model import TemplateModel

class CustomSwinTransformer(TemplateModel):
    def __init__(self, type, num_classes):
        super().__init__(type, num_classes)

    def load_pretrained(self):
        if self.type == "swin_v2_t":
            return torch.hub.load("pytorch/vision", "swin_v2_t", weights="IMAGENET1K_V1")
        elif self.type == "swin_v2_s":
            return torch.hub.load("pytorch/vision", "swin_v2_s", weights="IMAGENET1K_V1")
        elif self.type == "swin_v2_b":
            return torch.hub.load("pytorch/vision", "swin_v2_b", weights="IMAGENET1K_V1")
        else:
            raise RuntimeError(f"Type {self.type} is not supported for model SwinTransformer!")

    def remove_final_layer(self):
        self.pretrained_model.head = nn.Sequential()

    def get_num_features_of_pretrained_model(self):
        return self.pretrained_model.head.in_features

    def forward(self, x):
        return super().forward(x)

    def unfreeze_pretrained_layers(self):
        return super().unfreeze_pretrained_layers()