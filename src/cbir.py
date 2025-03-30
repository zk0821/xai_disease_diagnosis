import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from data.transforms_creator import TransformsCreator
from data.dataset_loader import DatasetLoader
from data.data_loader_creator import DataLoaderCreator
from utils.parameter_storage import ParameterStorage
from models.cnn.resnet_model import CustomResNet
from models.cnn.efficientnet_model import CustomEfficientNet
import faiss
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def main():
    parameter_storage = ParameterStorage(
        name="legendary-sweep-27",
        model_architecture="efficient_net",
        model_type="b2",
        dataset="HAM_10000",
        size=(400, 400),
        do_oversampling=False,
        do_class_weights=True,
        optimizer="adam",
        learning_rate=0.001,
        weight_decay=0,
        criterion="cross_entropy",
        scheduler="none",
        epochs=10,
        batch_size=1,
        solarize=0,
        saturation=0,
        brightness=0,
        contrast=0,
        sharpness=0,
        hue=0,
        posterization=0,
        rotation=0,
        erasing=0,
        affine=0,
        crop=0,
        gaussian_noise=0,
    )
    device = torch.device("cuda")
    # Transforms
    transforms_creator = TransformsCreator(parameter_storage)
    transforms_creator.create_transforms()
    # Load the dataset
    dataset_loader = DatasetLoader(parameter_storage, transforms=transforms_creator)
    # Create the data loaders
    data_loader_creator = DataLoaderCreator(parameter_storage, dataset_loader)
    data_loader_creator.create_dataloaders()
    # Load the model
    # res_net = CustomResNet("resnet152", dataset_loader.num_classes)
    # res_net = nn.DataParallel(res_net)
    # res_net.to(device)
    # load_specified_weights(res_net, "res_net", "resnet152", "colorful-sweep-8")
    model = CustomEfficientNet("b2", dataset_loader.num_classes)
    model = nn.DataParallel(model)
    model.to(device)
    load_specified_weights(model, "efficient_net", "b2", "legendary-sweep-27")
    # Extracting features
    image_paths = []
    descriptors = []
    with torch.no_grad():
        model.eval()
        for image, label, image_path in data_loader_creator.train_dataloader:
            result = pooling_output(image.to(device), model)
            descriptors.append(result.cpu().view(1, -1).numpy())
            image_paths.append(image_path)
            torch.cuda.empty_cache()
    print("Len Image Paths")
    print(len(image_paths))
    print("5 Image Paths")
    print(image_paths[:5])
    print("First Descriptor")
    print(descriptors[0])
    print("Shape of First Descriptor")
    print(descriptors[0].shape)
    # FAISS
    index = faiss.IndexFlatL2(2048)
    descriptors = np.vstack(descriptors)
    index.add(descriptors)
    # Query image
    query_image = "data/ham10000/train/images/ISIC_0024306.jpg"
    img = Image.open(query_image)
    input_tensor = transforms_creator.train_transforms(img)
    input_tensor = input_tensor.view(1, *input_tensor.shape)
    with torch.no_grad():
        query_descriptors = pooling_output(input_tensor.to(device), model).cpu().numpy()
        distance, indices = index.search(query_descriptors.reshape(1, 2048), 9)
    # Plot
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    for file_index, ax_i in zip(indices[0], np.array(ax).flatten()):
        ax_i.imshow(plt.imread(image_paths[file_index][0]))
        ax_i.set_title(image_paths[file_index][0])
    plt.savefig("cbir.png")


def pooling_output(x, model):
    for layer_name, layer in model.module.named_children():
        print("Layer name: " + layer_name)
        x = layer(x)
        if layer_name == "pretrained_model":
            break
    return x


def load_specified_weights(model, architecture, type, name):
    model.load_state_dict(torch.load(f"models/{architecture}/{type}/{name}.pth"))


if __name__ == "__main__":
    main()
