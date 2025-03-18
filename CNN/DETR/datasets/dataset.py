import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {
            'img': image, 
            'label': self.labels[idx]
        }

def get_ds(config):
    # Read CSV files
    train_df = pd.read_csv(os.path.join(config['TRAIN_DIR'].parent, 'Training_set.csv'))

    # Get unique labels and create label mapping
    all_labels = train_df['label'].unique()
    label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
    config['n_classes'] = len(label_to_idx)
    print(f"Found {len(label_to_idx)} unique classes")

    # Convert string labels to indices
    train_labels = [label_to_idx[label] for label in train_df['label']]
    
    # Create full paths for images
    train_paths = [os.path.join(config['TRAIN_DIR'], fname) for fname in train_df['filename']]

    # Split training data into train and validation
    train_paths, valid_paths, train_labels, valid_labels = train_test_split(
        train_paths, train_labels,
        test_size=config['test_size']/100.0,
        stratify=train_labels,
        random_state=config['seed']
    )

    # Create datasets
    train_data = CustomDataset(train_paths, train_labels)
    valid_data = CustomDataset(valid_paths, valid_labels)

    print(f"Dataset sizes - Train: {len(train_data)}, Validation: {len(valid_data)}")

    # Create data loaders
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    valid_loader = DataLoader(
        dataset=valid_data,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    ) 

    return train_loader, valid_loader