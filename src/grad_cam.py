import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import os
import random
from collections import OrderedDict
from PIL import Image
from utils.parameter_storage import ParameterStorage
from data.dataset_loader import DatasetLoader
from data.data_loader_creator import DataLoaderCreator
from models.cnn.efficientnet_model import CustomEfficientNet
from models.ensemble.ensemble import EnsembleModel
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def calculate_iou_binary_masks(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculates the Intersection over Union (IoU) for two binary masks.

    Args:
        mask1 (np.ndarray): The first binary mask (e.g., ground truth),
                            expected to contain boolean or 0/1 integer values.
        mask2 (np.ndarray): The second binary mask (e.g., prediction),
                            expected to contain boolean or 0/1 integer values.
                            Must have the same shape as mask1.

    Returns:
        float: The IoU score, a value between 0.0 and 1.0. Returns 1.0 if both masks are
               completely empty (no foreground pixels), and 0.0 if one is empty and the
               other is not, or if they have no overlap.

    Raises:
        ValueError: If the shapes of the input masks do not match.
    """
    if mask1.shape != mask2.shape:
        raise ValueError("Masks must have the same shape.")

    # Convert masks to boolean arrays for logical operations
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)

    # Calculate Intersection: pixels that are foreground in both masks
    # (True Positives)
    intersection = np.sum(mask1_bool & mask2_bool)

    # Calculate Union: pixels that are foreground in either mask
    # (True Positives + False Positives + False Negatives)
    union = np.sum(mask1_bool | mask2_bool)

    # Handle edge cases:
    if union == 0:
        # If both masks are completely empty (no foreground pixels),
        # IoU is typically considered 1.0 (perfect match of absence).
        return 1.0
    else:
        iou = intersection / union
        return iou


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
    with GradCAM(model=model, target_layers=target_layers) as cam:
        # for _, data in enumerate(data_loader_creator.test_dataloader):
        # Selected image
        selected_image = 0
        # Batch data
        #        images, labels, img_path = data
        data_path = "data/ham10000/train/images/"
        image_name = "ISIC_0024307"
        image_type = ".jpg"
        img_path = data_path + image_name + image_type
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = Image.fromarray(image)
        transformation = transforms.Compose([transforms.ToTensor()])
        image = transformation(image)
        images = torch.unsqueeze(image, 0)
        # GradCAM
        grayscale_cam = cam(input_tensor=images)
        img = images[selected_image, :]
        rgb_image = img.permute(1, 2, 0).numpy()
        rgb_image_to_save = (rgb_image * 255).astype(np.uint8)
        rgb_image_to_save = Image.fromarray(rgb_image_to_save)
        # Create structure to save files
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
        grayscale_image = grayscale_cam[selected_image, :]
        """
        grayscale_image[grayscale_image < 0.5] = 0
        grayscale_image[grayscale_image >= 0.5] = 1
        print("grayscale image:", grayscale_image)
        """
        grayscale = Image.fromarray(grayscale_image * 255)
        grayscale.convert("L").save(os.path.join(save, "grayscale_gradcam.png"))
        # Visualize
        visualization = show_cam_on_image(rgb_image, grayscale_image, use_rgb=True)
        cam_pil_image = Image.fromarray(visualization)
        cam_pil_image.save(os.path.join(save, "grad_cam_image.png"))
        """
        # Groundtruth segmentation
        segmentation_path = "data/isic2018/segmentation/groundtruth/ISIC_0012236_segmentation.png"
        segmentation_image = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)
        segmentation_image = cv2.resize(segmentation_image, (224, 224))
        print("segmentation_image:", segmentation_image)
        iou = calculate_iou_binary_masks(grayscale_image, segmentation_image)
        print("iou:", iou)
        """


if __name__ == "__main__":
    main()
