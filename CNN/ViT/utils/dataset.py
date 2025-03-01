import os
import glob
import numpy as np

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, file_list, label_list):
        self.file_list = file_list
        self.label_list = label_list

    def __len__(self):
        return len(self.file_list)
    
    def transform(self, img):
        img = np.asarray(img)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        else:
            img = np.transpose(img, (2, 0, 1))
        
        if (img>1).any():
            img = img / 255.0

        return img

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        label = self.label_list[idx]
        img = Image.open(img_path)
        img = self.transform(img)

        return {
            'img': torch.as_tensor(img).float().contiguous(), 
            'label': torch.as_tensor(label).long().contiguous()
        }


def get_ds(config):
    labels = []
    for folder in os.listdir(config.TRAIN_DIR):
        labels.append(folder.split('/')[-1])

    print(f"Found labels: {labels}")
    train_data = []
    test_data = []
    for cls in labels:
        train_data.append(glob.glob(os.path.join(config.TRAIN_DIR, f'{cls}/*.jpg')))
        test_data.append(glob.glob(os.path.join(config.TEST_DIR, f'{cls}/*.jpg')))
        print(f"Class {cls}: {len(train_data[-1])} train files, {len(test_data[-1])} test files")

    train_list = np.concatenate(train_data)
    test_list = np.concatenate(test_data)
    
    print(f"Total samples - Train: {len(train_list)}, Test: {len(test_list)}")  # Debug print

    train_labels = [os.path.basename(os.path.dirname(x)) for x in train_list]
    test_labels = [os.path.basename(os.path.dirname(x)) for x in test_list]
    train_labels = [int(x[1]) for x in train_labels]
    test_labels = [int(x[1]) for x in test_labels]
    train_list, valid_list, train_labels, valid_labels = train_test_split(
        train_list, train_labels,
        test_size=config.test_size/100.0,
        stratify=train_labels,
    )
    train_data = CustomDataset(train_list, train_labels)
    valid_data = CustomDataset(valid_list, valid_labels)
    test_data = CustomDataset(test_list, test_labels)

    train_loader = DataLoader(dataset = train_data, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset = valid_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(dataset = test_data, batch_size=config.batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader