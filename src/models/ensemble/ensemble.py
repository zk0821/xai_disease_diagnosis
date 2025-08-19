import torch
import torch.nn as nn
from collections import OrderedDict
from models.cnn.efficientnet_model import CustomEfficientNet
from models.cnn.convnext_model import CustomConvNext
from models.cnn.densenet_model import CustomDenseNet
from models.cnn.resnet_model import CustomResNet
from models.transformer.swin_transformer import CustomSwinTransformer
from models.transformer.vision_transformer import CustomVisionTransformer


class EnsembleModel(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.device = torch.device("cuda")
        # Efficient Net
        """
        self.efficient_net = CustomEfficientNet("b2", num_classes)
        self.load_specified_weights(
            self.efficient_net, "efficient_net", "b2", "warm-frost-3457-fold-0-fold-1-fold-2-fold-3"
        )
        """
        # Efficient Net V2
        self.efficient_net_v2 = CustomEfficientNet("v2_l", num_classes)
        self.load_specified_weights(
            self.efficient_net_v2, "efficient_net", "v2_l", "dainty-butterfly-3465-fold-0-fold-1-fold-2-fold-3"
        )
        """
        # Res Net
        self.res_net = CustomResNet("resnet152", num_classes)
        self.load_specified_weights(
            self.res_net, "res_net", "resnet152", "copper-donkey-3470-fold-0-fold-1-fold-2-fold-3"
        )
        # Res NeXt
        self.res_next = CustomResNet("resnext101_64", num_classes)
        self.load_specified_weights(
            self.res_next, "res_net", "resnext101_64", "earnest-smoke-3473-fold-0-fold-1-fold-2-fold-3"
        )
        # Dense Net
        # self.dense_net = CustomDenseNet("densenet161", num_classes)
        # self.load_specified_weights(self.dense_net, "dense_net", "densenet161", "fresh-snow-3475-fold-0-fold-1-fold-2")
        """
        # Conv NeXt
        self.conv_next = CustomConvNext("convnext_large", num_classes)
        self.load_specified_weights(
            self.conv_next, "conv_next", "convnext_large", "pleasant-plasma-3480-fold-0-fold-1-fold-2-fold-3"
        )
        # Swin Transformer
        self.swin_transformer = CustomSwinTransformer("swin_b", num_classes)
        self.load_specified_weights(
            self.swin_transformer, "swin_transformer", "swin_b", "usual-totem-3490-fold-0-fold-1-fold-2"
        )
        """
        # Swin Transformer V2
        self.swin_transformer_v2 = CustomSwinTransformer("swin_v2_s", num_classes)
        self.load_specified_weights(
            self.swin_transformer_v2,
            "swin_transformer",
            "swin_v2_s",
            "misunderstood-sunset-3492-fold-0-fold-1-fold-2-fold-3-fold-4",
        )
        # Vision Transformer
        self.vision_transformer = CustomVisionTransformer("vit_l_16", num_classes)
        self.load_specified_weights(
            self.vision_transformer,
            "vision_transformer",
            "vit_l_16",
            "neat-salad-3485-fold-0-fold-1-fold-2-fold-3",
        )
        """
        # Module List
        self.ensemble = nn.ModuleList([self.efficient_net_v2, self.conv_next, self.swin_transformer])
        # Classifier
        self.classifier = nn.Linear(num_classes * 3, num_classes)

    def remove_module_prefix(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v  # remove 'module.' prefix
            else:
                new_state_dict[k] = v
        return new_state_dict

    def load_specified_weights(self, model, architecture, type, name):
        checkpoint = torch.load(f"models/{architecture}/{type}/{name}.pth")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        state_dict = self.remove_module_prefix(state_dict)
        model.load_state_dict(state_dict)

    def forward(self, x):
        """
        # y1 = self.efficient_net(x)
        y2 = self.efficient_net_v2(x)
        # y3 = self.res_net(x)
        # y4 = self.res_next(x)
        y5 = self.dense_net(x)
        y6 = self.conv_next(x)
        y7 = self.swin_transformer(x)
        y8 = self.swin_transformer_v2(x)
        # y9 = self.vision_transformer(x)
        # y = torch.cat((y1, y2, y3, y4, y5, y6, y7, y8, y9), dim=1)
        y = torch.cat((y2, y5, y6, y7, y8), dim=1)
        """
        outputs = [model(x) for model in self.ensemble]
        concat = torch.cat(outputs, dim=1)
        out = self.classifier(concat)
        return out

    def get_concat(self, x):
        outputs = [model(x) for model in self.ensemble]
        concat = torch.cat(outputs, dim=1)
        return concat

    def unfreeze_pretrained_layers(self):
        # Do nothing
        print("Pre-trained layers should remain frozen")
