import torch
import torch.nn as nn
from torchvision.transforms import v2
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from models.gradcam.grad_cam import GradCamModel
import csv


def main():
    mean = [0.5325749, 0.37914315, 0.39562908]
    spread = [0.36953005, 0.2814345, 0.29810122]
    transform = v2.Compose(
        [
            v2.Resize((400, 400)),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean, spread),
        ]
    )
    # Model
    model = GradCamModel()
    model.eval()
    # read the image
    img_name = "ISIC_0024402"
    label = 6
    img_path = f"data/ham10000/train/images/{img_name}.jpg"
    image = Image.open(img_path).convert("RGB")
    print("Loaded the image...")
    transformed_image = transform(image)
    print("Transformed the image...")
    transformed_image = transformed_image.to("cuda")
    transformed_image = transformed_image.unsqueeze(0)

    # get the most likely prediction of the model
    pred = model(transformed_image)
    print("Got the predictions...")
    # get probabilites
    probs = nn.functional.softmax(pred, dim=1).detach().cpu().numpy()
    with open(f"gradcam/{img_name}/probabilites.csv", "w", newline="\n") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"])
        writer.writerow([probs[0][0], probs[0][1], probs[0][2], probs[0][3], probs[0][4], probs[0][5], probs[0][6]])
    # get the gradient of the output with respect to the parameters of the model
    pred[:, label].backward()

    # pull the gradients out of the model
    gradients = model.get_activations_gradient()
    print("Got the gradients...")

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = model.get_activations(transformed_image).detach()
    print("Got the activations...")

    # weight the channels by corresponding gradients
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap.detach().cpu(), 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    print("Created the heatmap...")

    # draw the heatmap
    plt.matshow(heatmap.squeeze())
    plt.savefig(f"gradcam/{img_name}/heatmap.jpg")
    img = cv2.imread(img_path)
    img = cv2.resize(img, (400, 400))
    cv2.imwrite(f"gradcam/{img_name}/original.jpg", img)
    heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite(f"gradcam/{img_name}/gradcam.jpg", superimposed_img)


if __name__ == "__main__":
    main()
