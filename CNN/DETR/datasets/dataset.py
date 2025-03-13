import os
from PIL import Image

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.image_paths = [img for img in os.listdir(data_path) if img.endswith(".jpg")]
        self.boxes = []
        self.labels = []
        
    def __len__(self):
        return len(self.image_paths)      
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        boxes = self.boxes[idx]
        labels = self.labels[idx]
        

