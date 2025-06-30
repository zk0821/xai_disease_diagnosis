import torch
import numpy as np
import cv2
import random
import math
from PIL import Image

g_grade_num = 10
g_replace_value = [128, 128, 128]
g_color_order = "RGB"


def blend(image1, image2, factor):
    """Blend image1 and image2 using 'factor'.

    Factor can be above 0.0.  A value of 0.0 means only image1 is used.
    A value of 1.0 means only image2 is used.  A value between 0.0 and
    1.0 means we linearly interpolate the pixel values between the two
    images.  A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values
    between 0 and 255.

    Args:
      image1: An image Tensor of type uint8.
      image2: An image Tensor of type uint8.
      factor: A floating point value above 0.0.

    Returns:
      A blended image Tensor of type uint8.
    """
    if factor == 0.0:
        return image1
    if factor == 1.0:
        return image2

    image1 = image1.float()
    image2 = image2.float()

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = image1 + scaled

    # Interpolate
    if factor > 0.0 and factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return temp.type(torch.uint8)

    # We need to clip and then cast.
    temp = torch.clamp(temp, 0.0, 255.0)
    return temp.type(torch.uint8)


def mixup(image, image_for_mixup, magnitude):
    """ mixup the corresponding pixels of image 1 and image 2. """
    _max = 0.2      # mixed intensity：0.0--0.2,
    _min = 0.0
    p = (_max - _min) / g_grade_num
    factor = random.uniform(_min, _min + p * magnitude) # magnitude is random
    height, width = image.shape[:2]
    image2 = cv2.resize(image_for_mixup.numpy(), (width, height))
    image2 = torch.from_numpy(np.array(image2, dtype=np.uint8))
    return blend(image, image2, factor)


def gaussian_noise(image, magnitude):
    """add Gaussian noise to the image."""
    _max = 0.2  # noise intensity：0.0--0.2,
    _min = 0.0
    p = (_max - _min) / g_grade_num
    factor = random.uniform(_min, _min + p * magnitude)  # magnitude is random
    size = tuple(image.shape)
    # rand = np.random.uniform(-50, 50, size)    # Random noise
    rand = np.random.normal(0, 50, size)  # Gaussian noise

    image1 = image.float().numpy() + rand * factor
    image1 = torch.from_numpy(image1)
    image1 = torch.clamp(image1, 0.0, 255.0)
    return image1.type(torch.uint8)


def cutout(image, magnitude):
    """Apply cutout (https://arxiv.org/abs/1708.04552) to image.

    This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
    a random location within `img`. The pixel values filled in will be of the
    value `replace`. The located where the mask will be applied is randomly
    chosen uniformly over the whole image.

    Args:
      image: An image Tensor of type uint8.

    Returns:
      An image Tensor that is of type uint8.
    """
    image_height = image.shape[0]
    image_width = image.shape[1]
    # the range of the size of cutout: 0--50 pixel
    _max = 50
    _min = 0
    p = (_max - _min) / g_grade_num
    factor = random.uniform(_min, _min + p * magnitude)  # magnitude is random
    pad_size = round(factor)
    replace = tuple(g_replace_value)

    # Sample the center location in the image where the zero mask will be applied.
    cutout_center_height = random.randint(int(image_height * 0.2), int(image_height * 0.8))
    cutout_center_width = random.randint(int(image_width * 0.2), int(image_width * 0.8))

    lower_pad = max(0, cutout_center_height - pad_size)
    upper_pad = min(image_height, cutout_center_height + pad_size)
    left_pad = max(0, cutout_center_width - pad_size)
    right_pad = min(image_width, cutout_center_width + pad_size)

    image[lower_pad:upper_pad, left_pad:right_pad, 0] = replace[0]
    image[lower_pad:upper_pad, left_pad:right_pad, 1] = replace[1]
    image[lower_pad:upper_pad, left_pad:right_pad, 2] = replace[2]
    return image


def solarize(image, magnitude):
    """For each pixel in the image, select the pixel
    if the value is less than the threshold.
    Otherwise, subtract 255 from the pixel."""
    _min = 128
    _max = 255
    p = (_min - _max) / g_grade_num
    factor = random.uniform(_max + p * magnitude, _max)  # magnitude is random
    threshold = round(factor)
    # threshold = random.randint(100, 250)
    return torch.where(image < threshold, image, 255 - image)


def saturation(image, magnitude):
    """change the saturation of the image"""
    _max = 0.4
    _min = 0.0
    p = (_max - _min) / g_grade_num
    factor = random.uniform(_min, _min + p * magnitude)  # magnitude is random
    index = random.sample([-1, 1], 1)[0]
    factor = 1.0 + index * factor
    gray_img = cv2.cvtColor(image.numpy(), cv2.COLOR_BGR2GRAY)
    degenerate = torch.from_numpy(np.array(gray_img, dtype=np.uint8))  # h×w
    degenerate = degenerate.unsqueeze(-1)  # h*w*1
    degenerate = degenerate.repeat(1, 1, 3)  # h×w×3
    return blend(degenerate, image, factor)


def contrast(image, magnitude):
    """change the contrast of the image"""
    _max = 0.4
    _min = 0.0
    p = (_max - _min) / g_grade_num
    factor = random.uniform(_min, _min + p * magnitude)  # magnitude is random
    index = random.sample([-1, 1], 1)[0]
    factor = 1.0 + index * factor
    # factor = random.uniform(0.6, 1.4)
    gray_img = cv2.cvtColor(image.numpy(), cv2.COLOR_BGR2GRAY)
    mean = np.mean(np.array(gray_img))  # get the mean value of the gray image.
    degenerate = torch.full(image.shape, mean).byte()
    return blend(degenerate, image, factor)


def brightness(image, magnitude):
    """change the brightness of the image"""
    _max = 0.4
    _min = 0.0
    p = (_max - _min) / g_grade_num
    factor = random.uniform(_min, _min + p * magnitude)  # magnitude is random
    index = random.sample([-1, 1], 1)[0]
    factor = 1.0 + index * factor
    degenerate = torch.zeros(image.shape).byte()
    return blend(degenerate, image, factor)


def scale(image, magnitude):
    """scale the image, and the width and height of the image are scaled in the same proportion.
    if it is reduced too much, the width or height may be smaller than the minimum input
    size required by the CNN model, and an error will occur.
    """
    _max = 0.2
    _min = 0.0
    p = (_max - _min) / g_grade_num
    factor = random.uniform(_min, _min + p * magnitude)  # magnitude is random
    index = random.sample([-1, 1], 1)[0]
    factor = 1.0 + index * factor
    height, width = image.shape[:2]
    new_height = round(height * factor)
    new_width = round(width * factor)

    img_scale = cv2.resize(image.numpy(), (new_width, new_height))
    img_scale = torch.from_numpy(np.array(img_scale, dtype=np.uint8))
    return torch.from_numpy(np.array(img_scale, dtype=np.uint8))


def scale_xy_diff(image, magnitude):
    """scale the image, and the width and height of the image are scaled in the different proportion.
    if it is reduced too much, the width or height may be smaller than the minimum input
    size required by the CNN model, and an error will occur.
    """
    _max = 0.2
    _min = 0.0
    p = (_max - _min) / g_grade_num
    factor = random.uniform(_min, _min + p * magnitude)  # magnitude is random
    index = random.sample([-1, 1], 1)[0]
    factor_x = 1.0 + index * factor
    factor_y = random.uniform(0.8, 1.2)
    height, width = image.shape[:2]
    new_height = round(height * factor_y)
    new_width = round(width * factor_x)

    img_scale = cv2.resize(image.numpy(), (new_width, new_height))
    img_scale = torch.from_numpy(np.array(img_scale, dtype=np.uint8))
    return torch.from_numpy(np.array(img_scale, dtype=np.uint8))


def shear_x(image, magnitude):
    """the image is sheared in the horizontal direction"""
    _max = 15.0
    _min = 0.0
    p = (_max - _min) / g_grade_num
    factor = random.uniform(_min, _min + p * magnitude)  # magnitude is random
    index = random.sample([-1, 1], 1)[0]
    degrees = index * factor
    replace = tuple(g_replace_value)
    height, width = image.shape[:2]

    degrees_to_radians = math.pi / 180.0
    radians = degrees * degrees_to_radians
    x_move = (height / 2) * math.tan(radians)
    points1 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    points2 = np.float32([[x_move, 0], [width + x_move, 0], [width - x_move, height], [-x_move, height]])
    matrix = cv2.getPerspectiveTransform(points1, points2)
    img_shear = cv2.warpPerspective(image.numpy(), matrix, (width, height), borderValue=replace)
    return torch.from_numpy(np.array(img_shear, dtype=np.uint8))


def shear_y(image, magnitude):
    """the image is sheared in the vertical direction"""
    _max = 15.0
    _min = 0.0
    p = (_max - _min) / g_grade_num
    factor = random.uniform(_min, _min + p * magnitude)  # magnitude is random
    index = random.sample([-1, 1], 1)[0]
    degrees = index * factor
    replace = tuple(g_replace_value)
    height, width = image.shape[:2]

    degrees_to_radians = math.pi / 180.0
    radians = degrees * degrees_to_radians
    y_move = (width / 2) * math.tan(radians)
    points1 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    points2 = np.float32([[0, y_move], [width, -y_move], [width, height - y_move], [0, height + y_move]])
    matrix = cv2.getPerspectiveTransform(points1, points2)
    img_shear = cv2.warpPerspective(image.numpy(), matrix, (width, height), borderValue=replace)
    return torch.from_numpy(np.array(img_shear, dtype=np.uint8))


def vignetting(image, magnitude):
    """Compared with the center of the image, darken the periphery of the image."""
    _max = 0.6
    _min = 0.0
    p = (_max - _min) / g_grade_num
    factor = random.uniform(_min, _min + p * magnitude)  # magnitude is random
    height, width = image.shape[:2]

    center_x = random.uniform(
        -width / 10, width / 10
    )  # randomly determine the position of the center point of the vignetting
    center_y = random.uniform(-height / 10, height / 10)
    min_dist = (
        width / 2.0 * np.random.uniform(0.3, 0.7)
    )  # the distance from the starting position of vignetting to the center point

    # creat matrix of distance from the center on the two axis
    x, y = np.meshgrid(
        np.linspace(-width / 2 + center_x, width / 2 + center_x, width),
        np.linspace(-height / 2 + center_y, height / 2 + center_y, height),
    )
    x, y = np.abs(x), np.abs(y)
    z = np.sqrt(x**2 + y**2)

    # creat the vignette mask on the two axis
    z = (z - min_dist) / (np.max(z) - min_dist)
    z = np.clip(z, 0, 1)
    z = z**1.2  # change to non-linear
    z = z * factor
    z = torch.from_numpy(z)  # h×w
    z = z.unsqueeze(-1)  # h*w*1
    z = z.repeat(1, 1, 3)  # h×w×3

    image = image.float()
    image = image * (1.0 - z)
    image = torch.clamp(image, 0.0, 255.0)
    return image.type(torch.uint8)


def lens_distortion(image, magnitude):
    """simulate lens distortion to transform the image."""
    d_coef = np.array([0.15, 0.15, 0.1, 0.1])  # k1, k2, p1, p2
    _max = 0.6
    _min = 0.0
    p = (_max - _min) / g_grade_num
    d_factor = np.random.uniform(_min, _min + p * magnitude, 4)  # magnitude is random
    d_factor = d_factor * (2 * (np.random.random(4) < 0.5) - 1)  # add sign
    d_coef = d_coef * d_factor

    height, width = image.shape[:2]
    # compute its diagonal
    f = (height**2 + width**2) ** 0.5
    # set the image projective to carrtesian dimension
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
    # Generate new camera matrix from parameters
    M, _ = cv2.getOptimalNewCameraMatrix(K, d_coef, (width, height), 0.5)
    # Generate look-up tables for remapping the camera image
    remap = cv2.initUndistortRectifyMap(K, d_coef, None, M, (width, height), cv2.CV_32FC2)
    # Remap the original image to a new image
    replace = tuple(g_replace_value)
    img = cv2.remap(image.numpy(), *remap, cv2.INTER_LINEAR, borderValue=replace)
    return torch.from_numpy(np.array(img, dtype=np.uint8))


def posterize(image, magnitude):
    """Equivalent of PIL Posterize. change the low n-bits of each pixel of the image to 0"""
    _min = 0.0
    _max = 3.0
    p = (_max - _min) / g_grade_num
    factor = random.uniform(_min, _min + p * magnitude)  # magnitude is random
    bits = round(factor)
    shift = bits
    img = image.byte() >> shift
    img = img << shift
    return img


def rotate(image, magnitude):
    """Rotates the image by degrees either clockwise or counterclockwise.

    Args:
      image: An image Tensor of type uint8.
      degrees: Float, a scalar angle in degrees to rotate all images by. If
        degrees is positive the image will be rotated clockwise otherwise it will
        be rotated counterclockwise.
      replace: A one or three value 1D tuple to fill empty pixels caused by
        the rotate operation.

    Returns:
      The rotated version of image.
    """
    _max = 40.0
    _min = 0.0
    p = (_max - _min) / g_grade_num
    factor = random.uniform(_min, _min + p * magnitude)  # magnitude is random
    index = random.sample([-1, 1], 1)[0]
    degrees = index * factor
    replace = tuple(g_replace_value)
    height, width = image.shape[:2]
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1)
    imgRotation = cv2.warpAffine(image.numpy(), matRotation, (width, height), borderValue=replace)
    return torch.from_numpy(np.array(imgRotation, dtype=np.uint8))


def autocontrast(image):
    """Implements Autocontrast function from PIL using TF ops.
    Args:
      image: A 3D uint8 tensor.

    Returns:
      The image after it has had autocontrast applied to it and will be of type
      uint8.
    """

    def scale_channel(image):
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = torch.min(image).float()
        hi = torch.max(image).float()

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = im.float() * scale + offset
            im = torch.clamp(im, 0.0, 255.0)
            return im.type(torch.uint8)

        return torch.where(hi > lo, scale_values(image), image)

    image = scale_channel(image)
    return image


def sharpness(image, magnitude):
    """Implements Sharpness function from PIL using TF ops."""
    _max = 0.4
    _min = 0.0
    p = (_max - _min) / g_grade_num
    factor = random.uniform(_min, _min + p * magnitude)  # magnitude is random
    index = random.sample([-1, 1], 1)[0]
    factor = 1.0 + index * factor
    orig_image = image
    image = image.float()
    # Make image 4D for conv operation.
    image = image.permute(2, 0, 1).contiguous()  # data from h*w*3 to 3*h*w
    image = image.unsqueeze(0)  # 1*3*h*w
    # smooth PIL Kernel.
    kernel = torch.tensor([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=torch.float32)  # 3*3
    kernel = kernel / 13.0  # normalize
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # 1*1*3*3
    kernel = kernel.repeat(3, 1, 1, 1)  # 3*1*3*3

    conv = torch.nn.Conv2d(3, 3, 3, groups=3, bias=False)
    conv.weight.data = kernel
    degenerate = conv(image)
    degenerate = torch.clamp(degenerate, 0, 255).byte().squeeze(0)
    degenerate = degenerate.permute(1, 2, 0).contiguous()  # data form 3×h×w to h×w×3

    # For the borders of the resulting image, fill in the values of the
    # original image.
    result = torch.zeros(orig_image.shape).byte()
    result[:, :, :] = orig_image
    result[1:-1, 1:-1, :] = degenerate
    return blend(result, orig_image, factor)


def equalize(image):
    """Implements Equalize function from PIL using TF ops.
    For each color channel, implements Equalize function.
    """

    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = im[:, :, c].numpy()
        inhist = cv2.equalizeHist(im)
        return torch.from_numpy(np.array(inhist, dtype=np.uint8))

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0).unsqueeze(-1)
    s2 = scale_channel(image, 1).unsqueeze(-1)
    s3 = scale_channel(image, 2).unsqueeze(-1)
    image = torch.cat([s1, s2, s3], 2)
    return image


def equalize_YUV(image):
    """Implements Equalize function from PIL using TF ops.
    Transforms the image to YUV color space, and then only implements Equalize function on the brightness Y
    """
    img = image.numpy()
    if g_color_order == "RGB":  # PIL format
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:  # Opencv format
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    channels_yuv = cv2.split(img_yuv)
    channels_yuv = list(channels_yuv)
    channels_yuv[0] = cv2.equalizeHist(channels_yuv[0])
    channels = cv2.merge(channels_yuv)
    if g_color_order == "RGB":  # PIL format
        result = cv2.cvtColor(channels, cv2.COLOR_YCrCb2RGB)
    else:  # Opencv format
        result = cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)
    return torch.from_numpy(np.array(result, dtype=np.uint8))


def flip(image):
    """Image is randomly flipped"""
    # factor: 0:vertical; 1:horizontal; -1:diagonal mirror
    factor = random.randint(-1, 1)
    img = image.numpy()
    img = cv2.flip(img, factor)
    return torch.from_numpy(np.array(img, dtype=np.uint8))


def crop(image, need_rand=True, nsize=(224, 224), rand_rate=(1.0, 1.0)):
    """random crop
    nsize: crop size
    need_rand: random crop or center crop
    rand_rate: The allowed region close to the center of the image for random cropping. (value: 0.7-1.0)
    """
    image_height = image.shape[0]
    image_width = image.shape[1]
    new_height = round(nsize[0])
    new_width = round(nsize[1])

    if image_width >= image_height:  # width is greater than height
        x_l = int(image_width * (1.0 - rand_rate[0]) / 2)
        x_r = int(image_width - x_l) - new_width
        y_l = int(image_height * (1.0 - rand_rate[1]) / 2)
        y_r = int(image_height - y_l) - new_height
    else:  # width is smaller than height
        x_l = int(image_width * (1.0 - rand_rate[1]) / 2)
        x_r = int(image_width - x_l) - new_width
        y_l = int(image_height * (1.0 - rand_rate[0]) / 2)
        y_r = int(image_height - y_l) - new_height
    if x_r <= x_l or y_r <= y_l:
        raise ValueError("Invalid rand_rate: {}".format(rand_rate))

    if 0 < new_height < image_height:
        if need_rand:
            start_h = random.randint(y_l, y_r)
        else:
            start_h = int((image_height - new_height) / 2)
    else:
        start_h = 0
        new_height = image_height
    if 0 < new_width < image_width:
        if need_rand:
            start_w = random.randint(x_l, x_r)
        else:
            start_w = int((image_width - new_width) / 2)
    else:
        start_w = 0
        new_width = image_width
    image = image[start_h : start_h + new_height, start_w : start_w + new_width, :]
    return image


def resize(image, min_size=256):
    """Resize the image to a fixed size, and keep the horizontal and vertical ratio unchanged
    min_size：the value to which the short side of the image is resized
    """
    image_height = image.shape[0]
    image_width = image.shape[1]

    if image_height < image_width:
        new_height = round(min_size)
        factor = min_size / image_height
        new_width = round(image_width * factor)
    else:
        new_width = round(min_size)
        factor = min_size / image_width
        new_height = round(image_height * factor)

    img_scale = cv2.resize(image.numpy(), (new_width, new_height))
    img_scale = torch.from_numpy(np.array(img_scale, dtype=np.uint8))
    return torch.from_numpy(np.array(img_scale, dtype=np.uint8))


def color_casting(image, magnitude):
    """Add a bias to a color channel in RGB
    For example, add a bias of 15 in the B color channel to each pixel, the image will be bluish.
    """
    prob_0 = random.randint(0, 2)
    for i in range(3):
        prob = random.randint(-1, 1)
        if prob_0 == i or prob == 1:
            _max = 30.0
            _min = 0.0
            p = (_max - _min) / g_grade_num
            factor = random.uniform(_min, _min + p * magnitude)  # magnitude is random
            index = random.sample([-1, 1], 1)[0]
            bias = index * factor
            img = image[:, :, i].float()
            img = img + bias
            img = torch.clamp(img, 0.0, 255.0).type(torch.uint8)
            image[:, :, i] = img
    return image


def multi_crop(image, num_crops=16):
    return Image.fromarray(image.numpy())
    crops = []
    for index in range(num_crops):
        print("multi crop shape:", image.shape)
        C, H, W = image.shape
        l_region = 1.0
        s_region = 0.8
        y_n = index // 4
        x_n = index % 4

        # Determine crop region size
        if W >= H:
            x_region = int(W * l_region)
            y_region = int(H * s_region)
        else:
            x_region = int(W * s_region)
            y_region = int(H * l_region)

        x_region = max(x_region, 224)
        y_region = max(y_region, 224)

        # Center the region
        x_cut = (W - x_region) // 2
        y_cut = (H - y_region) // 2

        # Compute crop top-left coordinates
        x_loc = x_cut + int(x_n * (x_region - 224) / (4 - 1))
        y_loc = y_cut + int(y_n * (y_region - 224) / (4 - 1))

        # Apply crop
        crop = image[:, y_loc : y_loc + 224, x_loc : x_loc + 224]
        crops.append(crop)
    return crops


class RandomCropInRate(object):
    """random crop
    nsize: crop size
    rand_rate: The allowed region close to the center of the image for random cropping. (value: 0.7-1.0)
    """

    def __init__(self, nsize, rand_rate=(1.0, 1.0)):
        self.nsize = nsize
        self.rand_rate = rand_rate  # rand_rate: (l, s)

    def __call__(self, image):
        image_height = image.size[1]
        image_width = image.size[0]
        new_height = self.nsize[0]
        new_width = self.nsize[1]

        if image_width >= image_height:
            x_l = int(image_width * (1.0 - self.rand_rate[0]) / 2)
            x_r = int(image_width - x_l) - new_width
            y_l = int(image_height * (1.0 - self.rand_rate[1]) / 2)
            y_r = int(image_height - y_l) - new_height
        else:
            x_l = int(image_width * (1.0 - self.rand_rate[1]) / 2)
            x_r = int(image_width - x_l) - new_width
            y_l = int(image_height * (1.0 - self.rand_rate[0]) / 2)
            y_r = int(image_height - y_l) - new_height
        if x_r <= x_l or y_r <= y_l:
            raise ValueError("Invalid rand_rate: {}".format(self.rand_rate))

        if 0 < new_height < image_height:
            start_h = random.randint(y_l, y_r)
        else:
            start_h = 0
            new_height = image_height
        if 0 < new_width < image_width:
            start_w = random.randint(x_l, x_r)
        else:
            start_w = 0
            new_width = image_width
        image = np.array(image)
        image = image[start_h : start_h + new_height, start_w : start_w + new_width, :]
        return Image.fromarray(image.astype(np.uint8))
