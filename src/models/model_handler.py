from models.cnn.resnet_model import CustomResNet
from models.cnn.densenet_model import CustomDenseNet
from models.cnn.efficientnet_model import CustomEfficientNet
from models.cnn.convnext_model import CustomConvNext
from models.transformer.swin_transformer import CustomSwinTransformer
from models.transformer.vision_transformer import CustomVisionTransformer
from models.ensemble.ensemble import EnsembleModel
from models.trainer import Trainer


class ModelHandler:

    def __init__(self, parameter_storage, evaluator):
        self.parameter_storage = parameter_storage
        self.evaluator = evaluator
        self.model = None
        self.trainer = None

    def prepare_model(self, dataset_loader, data_loader_creator):
        if self.parameter_storage.model_architecture == "res_net":
            print("Selected Model Architecture: ResNet")
            self.model = CustomResNet(self.parameter_storage.model_type, self.evaluator.num_classes)
        elif self.parameter_storage.model_architecture == "dense_net":
            print("Selected Model Architecture: DenseNet")
            self.model = CustomDenseNet(self.parameter_storage.model_type, self.evaluator.num_classes)
        elif self.parameter_storage.model_architecture == "efficient_net":
            print("Selected Model Architecture: EfficientNet")
            self.model = CustomEfficientNet(self.parameter_storage.model_type, self.evaluator.num_classes)
        elif self.parameter_storage.model_architecture == "conv_next":
            print("Selected Model Architecture: ConvNext")
            self.model = CustomConvNext(self.parameter_storage.model_type, self.evaluator.num_classes)
        elif self.parameter_storage.model_architecture == "swin_transformer":
            print("Selected Model Architecture: SwinTransformer")
            self.model = CustomSwinTransformer(self.parameter_storage.model_type, self.evaluator.num_classes)
        elif self.parameter_storage.model_architecture == "vision_transformer":
            print("Selected Model Architecture: VisionTransformer")
            self.model = CustomVisionTransformer(self.parameter_storage.model_type, self.evaluator.num_classes)
        elif self.parameter_storage.model_architecture == "ensemble":
            print("Selected Model Architecture: Ensemble")
            self.model = EnsembleModel(self.evaluator.num_classes)
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        else:
            raise RuntimeError(f'Model architecture"{self.parameter_storage.model}" is not supported!')
        self.trainer = Trainer(self.parameter_storage, self.model, dataset_loader, data_loader_creator, self.evaluator)
        self.trainer.prepare_model()

    def train_model(self, log_wandb=True, fold=None):
        assert self.trainer is not None
        self.trainer.train_model(log_wandb=log_wandb, fold=fold)

    def test_model(self, log_wandb=True, fold=None):
        assert self.trainer is not None
        self.trainer.load_model()
        self.trainer.test_model(log_wandb=log_wandb, fold=fold)