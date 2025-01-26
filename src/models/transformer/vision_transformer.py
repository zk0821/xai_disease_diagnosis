import torch
import torch.nn as nn

from models.template_model import TemplateModel

class CustomVisionTransformer(TemplateModel):
    def __init__(self, type, num_classes):
        super().__init__(type, num_classes)

    def load_pretrained(self):
        if self.type == "vit_b_16":
            return torch.hub.load(
                "pytorch/vision", "vit_b_16", weights="IMAGENET1K_SWAG_E2E_V1"
            )
        elif self.type == "vit_l_16":
            return torch.hub.load(
                "pytorch/vision", "vit_l_16", weights="IMAGENET1K_SWAG_E2E_V1"
            )
        else:
            raise RuntimeError(f"Type {self.type} is not supported for model VisionTransformer!")

    def remove_final_layer(self):
        self.pretrained_model.heads[-1] = nn.Sequential()

    def get_num_features_of_pretrained_model(self):
        print(f"Num in features: {self.pretrained_model.heads[-1].in_features}")
        return self.pretrained_model.heads[-1].in_features

    def forward(self, x):
        return super().forward(x)

    def unfreeze_pretrained_layers(self):
        super().unfreeze_pretrained_layers()