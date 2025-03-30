import torch
import torch.nn as nn
from models.cnn.resnet_model import CustomResNet


class GradCamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradients = None

        # PRETRAINED MODEL
        self.pretrained = CustomResNet("resnet152", 7)
        self.pretrained = nn.DataParallel(self.pretrained)
        self.pretrained.load_state_dict(torch.load(f"models/res_net/resnet152/colorful-sweep-8.pth"))

        self.features_conv = nn.Sequential(
            self.pretrained.module.pretrained_model.conv1,
            self.pretrained.module.pretrained_model.bn1,
            self.pretrained.module.pretrained_model.relu,
            self.pretrained.module.pretrained_model.maxpool,
            self.pretrained.module.pretrained_model.layer1,
            self.pretrained.module.pretrained_model.layer2,
            self.pretrained.module.pretrained_model.layer3,
            self.pretrained.module.pretrained_model.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            self.pretrained.module.flatten,
            self.pretrained.module.linear_one,
            self.pretrained.module.relu,
            self.pretrained.module.linear_two,
            self.pretrained.module.relu,
            self.pretrained.module.output,
        )

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # apply the remaining pooling
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)
