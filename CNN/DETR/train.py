import argparse

import torch
import torch.optim as optim
import torchvision.transforms as T

from CNN.DETR.model.DETR_simple import DETR

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--data-path", type=str, default="D:/MachineLearning/Datasets/VOC2012")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="ResNet50")
    parser.add_argument("--n-classes", type=int, default=20)
    parser.add_argument("--detr-pretrained", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    train_model(args)