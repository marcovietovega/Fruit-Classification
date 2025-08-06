"""
Fruit Classification Model Evaluation Script

This script evaluates a trained EfficientNetV2-based model for fruit classification.
Supports classification of cherries, strawberries, and tomatoes.

Author: Marco Vieto
Course: AIML421 - Victoria University of Wellington
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import csv
import sys
from typing import Tuple, List, Dict, Any


class FruitClassifier(nn.Module):
    """
    EfficientNetV2-based fruit classifier for cherry, strawberry, and tomato classification.

    Uses pre-trained EfficientNetV2-S as backbone with custom classifier head.
    """

    def __init__(self):
        super(FruitClassifier, self).__init__()

        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.efficient_net = efficientnet_v2_s(weights=weights)

        num_features = self.efficient_net.classifier[1].in_features

        self.efficient_net.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 3),
        )

    def forward(self, x):
        return self.efficient_net(x)


class FruitDataset(Dataset):
    """
    Custom PyTorch Dataset for fruit classification.

    Args:
        data_dir (str): Root directory containing class subdirectories
        transform (callable, optional): Optional transform to be applied on images
    """

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for class_idx, class_name in enumerate(["cherry", "tomato", "strawberry"]):
            class_path = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(class_path, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_indices(self):
        class_indices = {i: [] for i in range(3)}
        for idx, label in enumerate(self.labels):
            class_indices[label].append(idx)
        return class_indices


def validate_model_file(model_path: str) -> Tuple[bool, str]:
    """
    Validate if the model file exists and is properly formatted.

    Args:
        model_path (str): Path to the model file

    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        return False, f"Error: Model file '{model_path}' does not exist!"

    # Check file extension
    valid_extensions = [".pth", ".pt"]
    if not any(model_path.endswith(ext) for ext in valid_extensions):
        return (
            False,
            f"Error: Model file must have one of these extensions: {', '.join(valid_extensions)}",
        )

    # Check if file is empty
    if os.path.getsize(model_path) == 0:
        return False, f"Error: Model file '{model_path}' is empty!"

    return True, "Model file exists and appears valid"


def validate_folder_structure(base_path, required_classes):
    # Check if base folder exists
    if not os.path.exists(base_path):
        return False, f"Error: Base folder '{base_path}' does not exist!"

    if not os.path.isdir(base_path):
        return False, f"Error: '{base_path}' is not a directory!"

    # Check each required class folder
    missing_folders = []
    for class_name in required_classes:
        class_path = os.path.join(base_path, class_name)
        if not os.path.exists(class_path):
            missing_folders.append(class_name)
        elif not os.path.isdir(class_path):
            return False, f"Error: '{class_path}' exists but is not a directory!"

    if missing_folders:
        return (
            False,
            f"Error: Missing required class folders: {', '.join(missing_folders)}",
        )

    return True, "All required folders exist"


def verify_data_directory(data_dir, required_classes=None):
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")

    if required_classes is None:
        # Get all subdirectories in the data directory
        required_classes = [
            d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
        ]
        if not required_classes:
            raise ValueError(f"No class directories found in {data_dir}")

    # Verify each class directory
    total_images = 0
    for class_name in required_classes:
        class_path = os.path.join(data_dir, class_name)

        # Check if directory exists
        if not os.path.exists(class_path):
            raise ValueError(f"Class directory not found: {class_path}")

        # Check if it's actually a directory
        if not os.path.isdir(class_path):
            raise ValueError(f"Expected directory but found file: {class_path}")

        # Check for images
        image_files = [
            f
            for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not image_files:
            raise ValueError(f"No images found in class directory: {class_path}")

        total_images += len(image_files)
        print(f"✓ Verified {class_name} directory: {len(image_files)} images found")

    print(f"✓ Total images found: {total_images}")
    return True


def evaluate_predictions(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []
    misclassified = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            conf, predicted = torch.max(probabilities, 1)

            # Get filenames for this batch
            start_idx = batch_idx * test_loader.batch_size
            end_idx = start_idx + len(labels)
            batch_filenames = [
                os.path.basename(test_loader.dataset.images[i])
                for i in range(start_idx, end_idx)
            ]

            # Check for misclassifications
            for filename, pred, true_label, probability in zip(
                batch_filenames, predicted.cpu(), labels, conf.cpu()
            ):
                if pred != true_label:
                    misclassified.append(
                        {
                            "filename": filename,
                            "true_label": class_names[true_label],
                            "predicted": class_names[pred],
                            "probability": probability.item() * 100,
                        }
                    )

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confidences.extend(conf.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)

    # Calculate metrics
    conf_matrix = confusion_matrix(all_labels, all_preds)
    overall_accuracy = (all_preds == all_labels).mean() * 100
    per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1) * 100

    # Print results
    print("\nEvaluation Results:")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print("\nPer-class Accuracy:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name.capitalize()}: {per_class_accuracy[i]:.2f}%")

    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("\nConfusion Matrix:")
    print("Actual ↓\tPredicted →")
    header = "        \t" + "\t".join(f"{name:10}" for name in class_names)
    print(header)

    # Print each row with class name
    for i, class_name in enumerate(class_names):
        row = [f"{x:10}" for x in conf_matrix[i]]
        print(f"{class_name:8}\t" + "\t".join(row))

    # Print misclassified images
    if misclassified:
        print("\nMisclassified Images:")
        print(f"Total misclassified: {len(misclassified)}")

        with open("misclassified_images.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["filename", "true_label", "predicted", "probability"]
            )
            writer.writeheader()
            writer.writerows(misclassified)
        print(f"\nMisclassified images report saved to: misclassified_images.csv")


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = FruitClassifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def main():
    """
    Main execution function for model evaluation.

    Validates model and data, loads the trained model, and performs evaluation
    on the test dataset with comprehensive metrics reporting.
    """
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Paths
    MODEL_PATH = "./final_model.pth"
    TEST_DATA_PATH = "./testdata/"
    REQUIRED_CLASSES = ["cherry", "tomato", "strawberry"]

    # Validate model file first
    print("\nValidating model file...")
    is_valid, message = validate_model_file(MODEL_PATH)
    if not is_valid:
        print(message)
        print(
            f"\nPlease ensure the model file '{MODEL_PATH}' exists and is a valid PyTorch model file (.pth or .pt)"
        )
        sys.exit(1)
    else:
        print(message)

    # Validate folder structure first
    print("\nValidating folder structure...")
    is_valid, message = validate_folder_structure(TEST_DATA_PATH, REQUIRED_CLASSES)
    if not is_valid:
        print(message)
        print("\nRequired folder structure:")
        print(f"├── {TEST_DATA_PATH}")
        for class_name in REQUIRED_CLASSES:
            print(f"│   ├── {class_name}")
        print("\nPlease create the missing folders and add appropriate images.")
        sys.exit(1)
    else:
        print(message)

    # Verify test data directory contents
    try:
        verify_data_directory(TEST_DATA_PATH, REQUIRED_CLASSES)
        print("Data directory structure and contents verified successfully")
    except ValueError as e:
        print(f"Error in data directory structure: {e}")
        sys.exit(1)

    transform = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create test dataset and dataloader
    test_dataset = FruitDataset(TEST_DATA_PATH, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4 if device.type == "cuda" else 0,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Load model
    try:
        model = load_model(MODEL_PATH, device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("\nPlease ensure you have the correct model file and try again.")
        sys.exit(1)

    # Evaluate model
    results = evaluate_predictions(model, test_loader, device, REQUIRED_CLASSES)


if __name__ == "__main__":
    main()
