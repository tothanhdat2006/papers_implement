import os
import argparse
from pathlib import Path

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from ViT.ViT import VisionTransformer as ViT
from validation import run_validation
from configs.config import Config
from utils.dataset import get_ds

def train_model(config, device):
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")

    train_loader, valid_loader, test_loader = get_ds(config)
    model = ViT(img_sz=(256, 256), patches_sz=(16, 16), 
                num_classes=config.n_classes, n_layers=config.n_layers, 
                n_heads=config.n_heads, hid_dim=config.hid_dim, mlp_dim=config.mlp_dim, pool=config.pool).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    initial_epoch = 0
    global_step = 0

    if config.preload:
        if config.platform == "kaggle":
            model_filename = config.get_weight_file_path_kaggle("/kaggle/working/papers_implement/CNN/ViT", config.preload_name, config.preload)
        else:
            model_filename = config.get_weight_file_path(config.preload, config.preload_name)

        print(f'Preloading model {model_filename}')

        if torch.cuda.is_available():
            state = torch.load(model_filename, weights_only=True)
        else:
            state = torch.load(model_filename, map_location=torch.device("cpu"), weights_only=True)

        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    for epoch in range(initial_epoch, config.n_epochs):
        model.train()
        epoch_accuracy = 0.0
        epoch_loss = 0.0
        batch_iter = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.n_epochs}')
        for batch in batch_iter:
            img, label = batch['img'].to(device), batch['label'].to(device)

            output = model(img)

            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).sum().item() / label.size(0)
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)
            batch_iter.set_postfix({"loss": f"{loss.item():6.3f}", "accuracy": f"{acc:6.3f}"})

            global_step += 1

        loss, acc = run_validation(model, valid_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{config.n_epochs}, loss: {epoch_loss:.3f}, accuracy: {epoch_accuracy:.3f}")

    loss, acc = run_validation(model, test_loader, criterion, device)
    print(f"Test loss: {loss:.3f}, accuracy: {acc:.3f}")

    if config.save:
        if config.platform == "kaggle":
            model_filename = config.get_weight_file_path_kaggle("/kaggle/working/papers_implement/CNN/ViT", config.save_name, f'{epoch:02d}')
        else:
            model_filename = config.get_weight_file_path(config.save_name, f'{epoch:02d}')
        
        print(f'Saving model {model_filename}')

        if not os.path.exists(config.model_folder):
            os.makedirs(config.model_folder)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
    

def get_args():
    parser = argparse.ArgumentParser(description='Train Transformer')
    parser.add_argument('--n_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Number of epochs')
    parser.add_argument('--train_size', type=int, default=100, help='Training set size percentage')
    parser.add_argument('--n_classes', type=int, default=1000, help='Number of class')
    parser.add_argument('--save', type=bool, default=False, help='Save model or not')
    parser.add_argument('--save_name', type=str, default='ViT_weight_', help='Model weights name for saving')
    parser.add_argument('--preload', type=str, default=None, help="Weight's epoch to be loaded")
    parser.add_argument('--preload_name', type=str, default='ViT_weight_', help='Model weights name for preloading')
    parser.add_argument('--platform', type=str, default=None, help='Platform used to train')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    config = Config()
    config.n_classes = args.n_classes
    config.n_epochs = args.n_epochs
    config.train_size = args.train_size
    config.lr = args.lr
    config.platform = args.platform
    config.save = args.save
    config.save_name = args.save_name
    config.preload = args.preload
    config.preload_name = args.preload_name

    if config.platform == "kaggle":
        config.TRAIN_DIR = Path('/kaggle/working/papers_implement/CNN/ViT/data/train')
        config.TEST_DIR = Path('/kaggle/working/papers_implement/CNN/ViT/data/test')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_model(config, device)
    