import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Define data transformation pipeline
data_transforms = {
    'Train': transforms.Compose([
        transforms.Resize(600),  # Resize images to 333x333
          # Randomly crop images to 299x299
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize images
    ]),
    'Validation': transforms.Compose([
        transforms.Resize(600),
          # Center crop images to 299x299
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
}

# Define dataset paths
data_dir = '/content/extracted_files/Dataset'  # Change this to your dataset directory

# Create datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['Train', 'Validation']}

# Create data loaders
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['Train', 'Validation']}

# Get dataset sizes
dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Validation']}

# Get class names
class_names = image_datasets['Train'].classes

# Print class names
print("Class names:", class_names)
