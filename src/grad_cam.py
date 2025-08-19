import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import os
import glob
import random
from collections import OrderedDict
from PIL import Image
from utils.parameter_storage import ParameterStorage
from data.dataset_loader import DatasetLoader
from data.data_loader_creator import DataLoaderCreator
from models.cnn.efficientnet_model import CustomEfficientNet
from models.ensemble.ensemble import EnsembleModel
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM, XGradCAM, LayerCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import jaccard_score


def calculate_jaccard(groundtruth, prediction):
    # Ensure both are flattened and binary
    gt = np.array(groundtruth).flatten()
    pred = np.array(prediction).flatten()
    # If not binary, threshold
    gt = (gt > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)
    return jaccard_score(gt, pred, average="binary")


def get_gradcam(
    model, target_layers, image, grayscale_threshold=None, save_images=False, image_name="example", cam=None
):
    if cam is None:
        cam = GradCAM(model=model, target_layers=target_layers)
    images = torch.unsqueeze(image, 0)
    # GradCAM
    grayscale_cam = cam(input_tensor=images)
    img = images[0, :]
    rgb_image = img.permute(1, 2, 0).numpy()
    rgb_image_to_save = (rgb_image * 255).astype(np.uint8)
    rgb_image_to_save = Image.fromarray(rgb_image_to_save)
    # Create structure to save files
    if save_images:
        save_path = "grad_cam/"
        save = save_path + image_name
        if os.path.exists(save) and os.path.isdir(save):
            for filename in os.listdir(save):
                file_path = os.path.join(save, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            os.makedirs(save)
        rgb_image_to_save.save(os.path.join(save, "rgb_image.png"))
    grayscale_image = grayscale_cam[0, :]
    if grayscale_threshold is not None:
        grayscale_image[grayscale_image < grayscale_threshold] = 0
        grayscale_image[grayscale_image >= grayscale_threshold] = 1
    grayscale = Image.fromarray(grayscale_image * 255)
    if save_images:
        grayscale.convert("L").save(os.path.join(save, "grayscale_gradcam.png"))
    # Visualize
    visualization = show_cam_on_image(rgb_image, grayscale_image, use_rgb=True)
    cam_pil_image = Image.fromarray(visualization)
    if save_images:
        cam_pil_image.save(os.path.join(save, "grad_cam_image.png"))
    return grayscale


def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # remove 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict


def main():
    # Parameter storage
    parameter_storage = ParameterStorage(
        name="driven-silence-3518-fold-0-fold-1-fold-2-fold-3-fold-4",
        model_architecture="ensemble",
        model_type="all",
        dataset="HAM_10000",
        size=(224, 224),
        class_weights="balanced",
        weight_strategy="deferred",
        optimizer="adam",
        learning_rate=0.03,
        weight_decay=0,
        criterion="cross_entropy",
        scheduler="none",
        model_checkpoint=True,
        early_stoppage=False,
        epochs=50,
        batch_size=32,
        focal_loss_gamma=2,
        train_augmentation_policy="resize",
        train_augmentation_probability=0,
        train_augmentation_magnitude=0,
        test_augmentation_policy="resize",
        random_seed=2025,
    )
    # Set the random seeds for reproducibility
    torch.manual_seed(parameter_storage.random_seed)
    torch.cuda.manual_seed(parameter_storage.random_seed)
    torch.cuda.manual_seed_all(parameter_storage.random_seed)
    random.seed(parameter_storage.random_seed)
    np.random.seed(parameter_storage.random_seed)
    # Make GPUs deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Load the dataset
    dataset_loader = DatasetLoader(parameter_storage)
    # Create the data loaders
    data_loader_creator = DataLoaderCreator(parameter_storage, dataset_loader)
    data_loader_creator.create_dataloaders()
    # Create the model
    model = EnsembleModel(dataset_loader.num_classes)
    checkpoint = torch.load(
        f"models/{parameter_storage.model_architecture}/{parameter_storage.model_type}/{parameter_storage.name}.pth"
    )
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    target_layers = [
        model.efficient_net_v2.pretrained_model.features[-1],
        model.conv_next.pretrained_model.features[-1],
        model.swin_transformer.pretrained_model.features[-1],
    ]
    model.eval()
    # Data
    data_path = "data/ham10000/segmentation/"
    data_type = "test"
    if data_type == "train":
        print("Checking 'train' directory")
        # for_range = list(np.arange(0.1, 1, 0.1))
        for_range = [0.3]
    elif data_type == "test":
        print("Checking 'test' directory")
        for_range = [0.3]
    else:
        raise RuntimeError('f"Data type: {data_type}" is not supported!')
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    files = glob.glob(os.path.join(data_path, data_type, "images/*.jpg"))
    for groundtruth_threshold in for_range:
        jaccard_scores = []
        for index, file in enumerate(files):
            print(f"[{int(((index + 1) / len(files)) * 100)} %] File:", file)
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            image = Image.fromarray(image)
            transformation = transforms.Compose([transforms.ToTensor()])
            image = transformation(image)
            grayscale = get_gradcam(
                model,
                target_layers,
                image=image,
                grayscale_threshold=groundtruth_threshold,
                save_images=False,
                image_name=os.path.basename(file).replace(".jpg", ""),
                cam=cam,
            )
            # Groundtruth segmentation
            seg_name = os.path.basename(file).replace(".jpg", "_segmentation.png")
            segmentation_path = os.path.join(data_path, data_type, "groundtruth", seg_name)
            segmentation_image = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)
            segmentation_image = cv2.resize(segmentation_image, (224, 224))
            segmentation_image = (segmentation_image > 0).astype(np.uint8)
            # Jaccard index
            jaccard = calculate_jaccard(segmentation_image, grayscale)
            print("jaccard:", jaccard)
            jaccard_scores.append(jaccard)
        print(f"Threshold {groundtruth_threshold:.2f}: Mean Jaccard = {np.mean(jaccard_scores):.4f}")


if __name__ == "__main__":
    main()
