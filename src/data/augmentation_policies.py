import random
import cv2
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
    multi_crop,
)


def policy_multi_crop():
    return {0: [("Multi_crop", 1.0, 1.0)]}


def policy_v1_0(probability=0.7, magnitude=5):
    return {
        # color augment
        0: [
            [("Vignetting", probability, magnitude)],
            [("Gaussian_noise", probability, magnitude)],
            [("Saturation", probability, magnitude)],
            [("Contrast", probability, magnitude)],
            [("Brightness", probability, magnitude)],
            [("Sharpness", probability, magnitude)],
            [("Color_casting", probability, magnitude)],
            [("Equalize_YUV", probability, magnitude)],
            [("Posterize", probability, magnitude)],
            [("AutoContrast", probability, magnitude)],
            [("Solarize", probability, magnitude)],
            [("Equalize", probability, magnitude)],
        ],
        # shape augment
        1: [
            [("Rotate", probability, magnitude)],
            [("Lens_distortion", probability, magnitude)],
            [("Flip", probability, magnitude)],
            [("Cutout", probability, magnitude)],
            [("Shear_x", probability, magnitude)],
            [("Shear_y", probability, magnitude)],
            [("Scale", probability, magnitude)],
            [("Scale_xy_diff", probability, magnitude)],
        ],
    }


def augmentation_function(augmentation, image, magnitude):
    if augmentation == "AutoContrast":
        return autocontrast(image)
    elif augmentation == "Equalize":
        return equalize(image)
    elif augmentation == "Equalize_YUV":
        return equalize_YUV(image)
    elif augmentation == "Rotate":
        return rotate(image, magnitude)
    elif augmentation == "Shear_x":
        return shear_x(image, magnitude)
    elif augmentation == "Shear_y":
        return shear_y(image, magnitude)
    elif augmentation == "Scale":
        return scale(image, magnitude)
    elif augmentation == "Scale_xy_diff":
        return scale_xy_diff(image, magnitude)
    elif augmentation == "Posterize":
        return posterize(image, magnitude)
    elif augmentation == "Solarize":
        return solarize(image, magnitude)
    elif augmentation == "Saturation":
        return saturation(image, magnitude)
    elif augmentation == "Contrast":
        return contrast(image, magnitude)
    elif augmentation == "Brightness":
        return brightness(image, magnitude)
    elif augmentation == "Sharpness":
        return sharpness(image, magnitude)
    elif augmentation == "Cutout":
        return cutout(image, magnitude)
    elif augmentation == "Gaussian_noise":
        return gaussian_noise(image, magnitude)
    elif augmentation == "Vignetting":
        return vignetting(image, magnitude)
    elif augmentation == "Lens_distortion":
        return lens_distortion(image, magnitude)
    elif augmentation == "Flip":
        return flip(image)
    elif augmentation == "Crop":
        return crop(image, magnitude)
    elif augmentation == "Color_casting":
        return color_casting(image, magnitude)
    elif augmentation == "Resize":
        return resize(image, magnitude)
    elif augmentation == "Multi_crop":
        return multi_crop(image, num_crops=16)
    else:
        raise RuntimeError(f"Unsupported augmentation: {augmentation}")


def apply_policy(policy, image):
    if policy == "v1_0":
        selected_policy = policy_v1_0()
    elif policy == "multi_crop":
        selected_policy = policy_multi_crop()
    else:
        raise RuntimeError("Unsupported policy: {policy}")
    augmented_image = image
    for augmentation_category in selected_policy.values():
        picked_augmentation = random.choice(augmentation_category)
        augmentation, probability, magnitude = picked_augmentation[0]
        if random.random() < probability:
            augmented_image = augmentation_function(augmentation, augmented_image, magnitude)
    return augmented_image
