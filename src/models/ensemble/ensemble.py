import torch
import torch.nn as nn
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
        self.efficient_net = CustomEfficientNet("b2", num_classes)
        self.efficient_net = nn.DataParallel(self.efficient_net)
        self.efficient_net.to(self.device)
        self.load_specified_weights(self.efficient_net, "efficient_net", "b2", "serene-moon-2936")
        self.dense_net = CustomDenseNet("densenet201", num_classes)
        self.dense_net= nn.DataParallel(self.dense_net)
        self.dense_net.to(self.device)
        self.load_specified_weights(self.dense_net, "dense_net", "densenet201", "avid-wildflower-2954")
        self.swin_transformer = CustomSwinTransformer("swin_v2_t", num_classes)
        self.swin_transformer = nn.DataParallel(self.swin_transformer)
        self.swin_transformer.to(self.device)
        self.load_specified_weights(self.swin_transformer, "swin_transformer", "swin_v2_t", "rose-wave-2949")
        self.conv_next = CustomConvNext("convnext_base", num_classes)
        self.conv_next = nn.DataParallel(self.conv_next)
        self.conv_next.to(self.device)
        self.load_specified_weights(self.conv_next, "conv_next", "convnext_base", "scarlet-blaze-2947")
        self.res_net = CustomResNet("resnext50", num_classes)
        self.res_net= nn.DataParallel(self.res_net)
        self.res_net.to(self.device)
        self.load_specified_weights(self.res_net, "res_net", "resnext50", "neat-forest-2959")
        self.classifier = nn.Linear(num_classes * 5, num_classes)

    def load_specified_weights(self, model, architecture, type, name):
        model.load_state_dict(
            torch.load(
                f"models/{architecture}/{type}/{name}.pth"
            )
        )

    def forward(self, x):
        y1 = self.efficient_net(x)
        #y2 = self.efficient_net_v2(x)
        y3 = self.dense_net(x)
        y4 = self.swin_transformer(x)
        #y5 = self.vision_transformer(x)
        y6 = self.conv_next(x)
        y7 = self.res_net(x)
        y = torch.cat((y1, y3, y4, y6, y7), dim=1)
        out = self.classifier(y)
        return out

    def unfreeze_pretrained_layers(self):
        # Do nothing
        print("Pre-trained layers should remain frozen")