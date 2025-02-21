import os
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, filepaths: list, labels: list, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        filepath = self.filepaths[idx]
        label = self.labels[idx]
        image = Image.open(filepath)

        if self.transform:
            image = self.transform(image)

        return {
            'image': torch.as_tensor(image.copy()).float().contiguous(),
            'label': torch.as_tensor(label).long().contiguous()
        }

def transform(img, img_size=(224, 224)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(img_size, resample=Image.BICUBIC)
    img = np.asarray(img)

    img = img.transpose((2, 0, 1))  # Convert HWC to CHW format
    
    if (img > 1).any():
        img = img / 255.0  # Normalize to [0,1]
    
    return img

def load_data(dir_data: str, val_percent: float = 0.2, batch_size: int = 1, num_workers: int = 0):
    '''
    Load data from directory structure:
    dir_data/
        train/
            class1/
                img1.*
                img2.*
            class2/
                img1.*
                img2.*
            ...

    Args:
        dir_data (str): Path to the data directory
        val_percent (float): Fraction of data to use for validation
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of workers for DataLoader

    Returns:
        dict: Mapping from class names to indices
        DataLoader: Training data loader
        DataLoader: Validation data loader
    '''
    train_dir = Path(dir_data) / 'train'
    filepaths = []
    labels = []
    
    # Get class folders
    class_folders = [cls for cls in train_dir.iterdir() if cls.is_dir()]
    class_to_idx = {folder.name: idx for idx, folder in enumerate(sorted(class_folders))}
    
    # Collect all image paths and their labels
    for class_folder in class_folders:
        class_label = class_to_idx[class_folder.name]
        for img_path in class_folder.glob('*'):
            if img_path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff'):
                filepaths.append(str(img_path))
                labels.append(class_label)

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        filepaths, labels, 
        test_size=val_percent, 
        random_state=86, 
        stratify=labels
    )

    # Create datasets
    train_dataset = CustomDataset(X_train, y_train, transform=transform)
    val_dataset = CustomDataset(X_val, y_val, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    return class_to_idx, train_loader, val_loader