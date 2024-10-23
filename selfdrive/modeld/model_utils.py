"""
model_utils.py

This module contains utility functions and classes for the parking vehicle detection model.

Functions:
    load_model: Loads and configures the ResNet-18 model for binary classification.
    get_transform: Returns the image transformation pipeline.

Dependencies:
    - torch
    - torchvision
"""
import torch
from torchvision import transforms
from torchvision.models import resnet18

def load_model(model_path):
    """
    Loads and configures the ResNet-18 model for binary classification.

    Args:
        model_path (str): Path to the saved model weights.

    Returns:
        torch.nn.Module: Configured ResNet-18 model.
    """
    model = resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_transform(train=False):
    """
    Returns the image transformation pipeline.

    Args:
        train (bool): If True, includes data augmentation transforms.

    Returns:
        torchvision.transforms.Compose: Composition of image transforms.
    """
    base_transforms = [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    if train:
        train_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ]
        return transforms.Compose(train_transforms + base_transforms)

    return transforms.Compose(base_transforms)
