import torch
import torch.nn as nn

from models.template_model import TemplateModel

class CustomEfficientNet(TemplateModel):
    def __init__(self, type, num_classes):
        super().__init__(type, num_classes)

    def load_pretrained(self):
        if self.type == "b0":
            return torch.hub.load(
                "pytorch/vision", "efficientnet_b0", weights="IMAGENET1K_V1"
            )
        elif self.type == "b2":
            return torch.hub.load(
                "pytorch/vision", "efficientnet_b2", weights="IMAGENET1K_V1"
            )
        elif self.type == "b4":
            return torch.hub.load(
                "pytorch/vision", "efficientnet_b4", weights="IMAGENET1K_V1"
            )
        elif self.type == "b5":
            return torch.hub.load(
                "pytorch/vision", "efficientnet_b5", weights="IMAGENET1K_V1"
            )
        elif self.type == "b7":
            return torch.hub.load(
                "pytorch/vision", "efficientnet_b7", weights="IMAGENET1K_V1"
            )
        elif self.type == "v2_s":
            return torch.hub.load(
                "pytorch/vision", "efficientnet_v2_s", weights="IMAGENET1K_V1"
            )
        elif self.type == "v2_l":
            return torch.hub.load(
                "pytorch/vision", "efficientnet_v2_l", weights="IMAGENET1K_V1"
            )
        else:
            raise RuntimeError(
                'Type "{self.type}" is not supported for model EfficientNet!'
            )

    def remove_final_layer(self):
        self.pretrained_model.classifier[1] = nn.Sequential()

    def get_num_features_of_pretrained_model(self):
        return self.pretrained_model.classifier[1].in_features

    def forward(self, x):
        return super().forward(x)

    def unfreeze_pretrained_layers(self):
        super().unfreeze_pretrained_layers()
