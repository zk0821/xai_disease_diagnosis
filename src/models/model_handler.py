from models.cnn.resnet_model import CustomResNet
from models.cnn.densenet_model import CustomDenseNet
from models.cnn.efficientnet_model import CustomEfficientNet
from models.cnn.convnext_model import CustomConvNext
from models.transformer.swin_transformer import CustomSwinTransformer
from models.transformer.vision_transformer import CustomVisionTransformer
from models.trainer import Trainer


class ModelHandler:

    def __init__(self, parameter_storage, evaluator):
        self.parameter_storage = parameter_storage
        self.evaluator = evaluator
        self.model = None
        self.transfer_learning = None

    def prepare_model(self, dataset_loader, data_loader_creator):
        if self.parameter_storage.model_architecture == "res_net":
            print("Selected Model Type: ResNet")
            self.model = CustomResNet(
                self.parameter_storage.model_type, self.evaluator.num_classes
            )
        elif self.parameter_storage.model_architecture == "dense_net":
            print("Selected Model Type: DenseNet")
            self.model = CustomDenseNet(
                self.parameter_storage.model_type, self.evaluator.num_classes
            )
        elif self.parameter_storage.model_architecture == "efficient_net":
            print("Selected Model Type: EfficientNet")
            self.model = CustomEfficientNet(
                self.parameter_storage.model_type, self.evaluator.num_classes
            )
        elif self.parameter_storage.model_architecture == "conv_next":
            print("Selected Model Type: ConvNext")
            self.model = CustomConvNext(
                self.parameter_storage.model_type, self.evaluator.num_classes
            )
        elif self.parameter_storage.model_architecture == "swin_transformer":
            print("Selected Model Type: SwinTransformer")
            self.model = CustomSwinTransformer(self.parameter_storage.model_type, self.evaluator.num_classes)
        elif self.parameter_storage.model_architecture == "vision_transformer":
            print("Selected Model Type: VisionTransformer")
            self.model = CustomVisionTransformer(self.parameter_storage.model_type, self.evaluator.num_classes)
        else:
            raise RuntimeError(
                f'Model architecture"{self.parameter_storage.model}" is not supported!'
            )
        self.transfer_learning = Trainer(
            self.parameter_storage, self.model, dataset_loader, data_loader_creator, self.evaluator
        )
        self.transfer_learning.prepare_model()

    def train_model(self):
        self.transfer_learning.train_model()

    def test_model(self):
        self.transfer_learning.load_model()
        self.transfer_learning.test_model()
