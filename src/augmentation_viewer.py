import torch
import cv2
import numpy as np
import os
from data.augmentation_functions import (
    autocontrast,
    equalize,
    equalize_YUV,
    rotate,
    shear_x,
    shear_y,
    scale,
    scale_xy_diff,
    posterize,
    solarize,
    saturation,
    contrast,
    brightness,
    sharpness,
    cutout,
    gaussian_noise,
    vignetting,
    lens_distortion,
    flip,
    crop,
    color_casting,
    resize,
    # multi_crop,
    mixup,
    full_resize,
)
from PIL import Image


def main():
    # Read the image
    img_path = "data/ham10000/train/images/ISIC_0024306.jpg"
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(np.array(image, dtype=np.uint8))
    # define augmentation
    image_for_mixup = "data/ham10000/train/images/ISIC_0024309.jpg"
    mixup_image = cv2.imread(image_for_mixup)
    mixup_image = cv2.cvtColor(mixup_image, cv2.COLOR_BGR2RGB)
    mixup_image = torch.from_numpy(np.array(mixup_image, dtype=np.uint8))
    augmentation = shear_y(image, 10)
    # Convert tensors back to PIL images for saving
    pil_image = Image.fromarray(image.numpy())
    pil_augmented = Image.fromarray(augmentation.numpy())

    # Prepare output directory
    out_dir = "augmentation_viewer"
    os.makedirs(out_dir, exist_ok=True)

    # Prepare output paths
    base_name = os.path.basename(img_path).replace(".jpg", "")
    orig_path = os.path.join(out_dir, f"{base_name}_orig.jpg")
    aug_path = os.path.join(out_dir, f"{base_name}_aug.jpg")

    # Save images
    pil_image.save(orig_path)
    pil_augmented.save(aug_path)
    print(f"Saved original to {orig_path}")
    print(f"Saved augmented to {aug_path}")


if __name__ == "__main__":
    main()
