import torch
import torch.nn as nn
from data.dataset_loader import DatasetLoader
from data.data_loader_creator import DataLoaderCreator
from utils.parameter_storage import ParameterStorage
from models.cnn.resnet_model import CustomResNet
from models.cnn.efficientnet_model import CustomEfficientNet
from models.ensemble.ensemble import EnsembleModel
import faiss
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import os


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
    # Number of retrieved images
    N = 20
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
    model.eval()
    # Extracting features
    image_paths = []
    descriptors = []
    for batch_images, batch_labels, batch_image_paths in data_loader_creator.test_dataloader:
        for ix, img in enumerate(batch_images):
            images = torch.unsqueeze(img, 0)
            result = model.get_concat(images)
            descriptors.append(result.detach().cpu().view(1, -1).numpy())
            image_paths.append(batch_image_paths[ix])
            torch.cuda.empty_cache()
        break
    # FAISS
    index = faiss.IndexFlatL2(21)
    descriptors = np.vstack(descriptors)
    index.add(descriptors)
    # Collections for all query images
    all_precisions = []
    all_recalls = []
    all_f1s = []
    # Query image
    batch = 0
    for batch_query_image, batch_query_label, batch_query_image_path in data_loader_creator.test_dataloader:
        batch += 1
        print(f"[Batch {batch}] Calculating...")
        for ix, img in enumerate(batch_query_image):
            query_image = batch_query_image[ix]
            query_label = batch_query_label[ix].item()
            query_image_path = batch_query_image_path[ix]
            query_basename = query_image_path.replace(".jpg", "")
            images = torch.unsqueeze(query_image, 0)
            query_descriptors = model.get_concat(images)

            # Get the set of all relevant images in the dataset (excluding the query itself)
            relevant_images = set(
                dataset_loader.test_dataframe.metadata_df[
                    dataset_loader.test_dataframe.metadata_df["type"] == query_label
                ]["image"].values
            )
            relevant_images.discard(query_basename)  # Remove the query image itself if present

            # Retrieve similar images
            retrieved_images = []
            distance, indices = index.search(query_descriptors.detach().cpu().reshape(1, 21), N)
            for file_index in indices[0]:
                # Get the image path for the retrieved file
                retrieved_image_path = (
                    image_paths[file_index][0]
                    if isinstance(image_paths[file_index], (list, tuple))
                    else image_paths[file_index]
                )
                # Get the basename
                basename = os.path.basename(retrieved_image_path).replace(".jpg", "")
                # Get the label from the DataFrame
                label = dataset_loader.test_dataframe.metadata_df.loc[
                    dataset_loader.test_dataframe.metadata_df["image"] == basename, "type"
                ].values
                label = label[0] if len(label) > 0 else "Unknown"
                if basename == query_basename:
                    print("QUERY WAS RETURNED!")
                else:
                    retrieved_images.append(basename)
            # Calculate number of relevant retrieved images
            relevant_retrieved = [img for img in retrieved_images if img in relevant_images]
            num_relevant_retrieved = len(relevant_retrieved)
            print("Num relevant retrieved:", num_relevant_retrieved)
            num_relevant = len(relevant_images)
            print("Num relevant:", num_relevant)
            num_retrieved = len(retrieved_images)
            print("Num retrieved:", num_retrieved)
            # Precision, Recall, F1
            precision = num_relevant_retrieved / num_retrieved if num_retrieved > 0 else 0
            print("Precision:", precision)
            all_precisions.append(precision)
            recall = num_relevant_retrieved / num_relevant if num_relevant > 0 else 0
            print("Recall:", recall)
            all_recalls.append(recall)
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            print("F1:", f1)
            all_f1s.append(f1)
    # Print the results
    mean_precision = sum(all_precisions) / len(all_precisions)
    mean_recall = sum(all_recalls) / len(all_recalls)
    mean_f1 = sum(all_f1s) / len(all_f1s)

    print(f"Mean Precision@{N}: {mean_precision:.4f}")
    print(f"Mean Recall@{N}: {mean_recall:.4f}")
    print(f"Mean F1@{N}: {mean_f1:.4f}")
    """
    # Plot
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    for file_index, ax_i in zip(indices[0], np.array(ax).flatten()):
        ax_i.imshow(plt.imread(image_paths[file_index][0]))
        ax_i.set_title(image_paths[file_index][0])
    plt.savefig("cbir.png")
    """


if __name__ == "__main__":
    main()
