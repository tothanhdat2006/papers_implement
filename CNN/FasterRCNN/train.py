import argparse

import torch
import torch.nn as nn
import torch.optim as optim

def get_args():
    parser = argparse.ArgumentParser(description="Training Faster R-CNN")
    parser.add_argument("--backbone", type=str, default="VGG16", help="Backbone list: {VGG16, SPPnet}")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    