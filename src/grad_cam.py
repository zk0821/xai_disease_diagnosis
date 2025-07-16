import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import random
from PIL import Image
from utils.parameter_storage import ParameterStorage
from data.dataset_loader import DatasetLoader
from data.data_loader_creator import DataLoaderCreator
from models.cnn.efficientnet_model import CustomEfficientNet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


global_large_dummy_tensor = None

def fake_memory_alloc(gb):
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot allocate GPU memory.")
        return
    # Calculate the number of elements needed for the desired memory amount
    # Assuming float32 (4 bytes per element)
    bytes_per_element = 4
    total_bytes = gb * 1024 * 1024 * 1024
    num_elements = total_bytes // bytes_per_element

    try:
        # Create a single large random tensor on the GPU
        dummy_tensor = torch.empty((int(num_elements),), device="cuda", dtype=torch.float32)
        # Initialize it to ensure memory is actually "used" by touching all pages
        dummy_tensor.uniform_()
        allocated_mb = dummy_tensor.numel() * bytes_per_element / (1024 * 1024)
        print(f"Allocated {allocated_mb:.2f} MB of GPU memory with a single tensor.")
        return dummy_tensor
    except RuntimeError as e:
        print(f"Failed to allocate {gb} GB of GPU memory: {e}")
        print("Consider reducing the amount or checking available VRAM.")
        return None

def print_gpu_usage():
    if torch.cuda.is_available():
        print(f"Current GPU memory allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        print(f"Max GPU memory allocated: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")

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

def main():
    print_gpu_usage()
    global_large_dummy_tensor = fake_memory_alloc(10)
    print_gpu_usage()
    # Parameter storage
    parameter_storage = ParameterStorage(
        name="Test GradCAM",
        model_architecture="efficient_net",
        model_type="b2",
        dataset="HAM_10000",
        size=(224, 224),
        do_oversampling=False,
        class_weights="none",
        optimizer="adam",
        learning_rate=2e-4,
        weight_decay=0,
        criterion="cross_entropy",
        scheduler='none',
        model_checkpoint=True,
        early_stoppage=False,
        epochs=70,
        batch_size=64,
        focal_loss_gamma=2,
        class_balance_beta=0.999,
        validation_split=0.2,
        train_augmentation_policy="resize",
        train_augmentation_probability=0,
        train_augmentation_magnitude=0,
        test_augmentation_policy="resize",
        random_seed=380
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
    model = CustomEfficientNet("b2", dataset_loader.num_classes)
    target_layers = [model.pretrained_model.features[-1]]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        #for _, data in enumerate(data_loader_creator.test_dataloader):
        # Selected image
        selected_image = 0
        # Batch data
#        images, labels, img_path = data
        img_path = "data/isic2018/segmentation/images/ISIC_0012236.jpg"
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = Image.fromarray(image)
        transformation = transforms.Compose([transforms.ToTensor()])
        image = transformation(image)
        images = torch.unsqueeze(image, 0)
        print("image:", image.shape)
       # print("Selected image:", img_path[selected_image])
       # print("images shape:", images.shape)
        # GradCAM
        grayscale_cam = cam(input_tensor=images)
        img = images[selected_image, :]
        print("image shape", img.shape)
        rgb_image = img.permute(1, 2, 0).numpy()
        print("rgb image:", rgb_image)
        rgb_image_to_save = (rgb_image * 255).astype(np.uint8)
        print("rgb image to save:", rgb_image_to_save)
        rgb_image_to_save = Image.fromarray(rgb_image_to_save)
        rgb_image_to_save.save("test_image.png")
        grayscale_image = grayscale_cam[selected_image, :]
        grayscale_image[grayscale_image < 0.5] = 0
        grayscale_image[grayscale_image >= 0.5] = 1
        print("grayscale image:", grayscale_image)
        grayscale = Image.fromarray(grayscale_image * 255)
        grayscale.convert("L").save("grayscale_gradcam.png")
        # Visualize
        visualization = show_cam_on_image(rgb_image, grayscale_image, use_rgb=True)
        cam_pil_image = Image.fromarray(visualization)
        cam_pil_image.save("grad_cam_image.png")
        # Groundtruth segmentation
        segmentation_path = "data/isic2018/segmentation/groundtruth/ISIC_0012236_segmentation.png"
        segmentation_image = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)
        segmentation_image = cv2.resize(segmentation_image, (224, 224))
        print("segmentation_image:", segmentation_image)
        iou = calculate_iou_binary_masks(grayscale_image, segmentation_image)
        print("iou:", iou)


if __name__ == "__main__":
    main()
