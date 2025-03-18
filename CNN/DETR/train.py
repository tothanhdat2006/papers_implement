import argparse
import os
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR
from model.DETR import build_DETR
from config.config import get_config, get_weight_file_path_kaggle, get_weight_file_path
from datasets.dataset import get_ds
from validation import run_validation

def train_model(config):
    print(f'Using device: {config['device']}')
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(config['device'].index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(config['device'].index).total_memory / 1024 ** 3} GB")

    train_loader, valid_loader = get_ds(config)
    model = build_DETR(config).to(config['device'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = StepLR(optimizer, step_size=1, gamma=config['gamma'])
    initial_epoch = 0
    global_step = 0

    if config['preload']:
        if config['platform'] == "kaggle":
            model_filename = get_weight_file_path_kaggle("/kaggle/working/papers_implement/CNN/DETR", config['preload_name'], config['preload'])
        else:
            model_filename = get_weight_file_path(config['preload'], config['preload_name'])

        print(f'Preloading model {model_filename}')

        if torch.cuda.is_available():
            state = torch.load(model_filename, weights_only=True)
        else:
            state = torch.load(model_filename, map_location=torch.device("cpu"), weights_only=True)

        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    for epoch in range(initial_epoch, config['n_epochs']):
        model.train()
        epoch_accuracy = 0.0
        epoch_loss = 0.0
        batch_iter = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config['n_epochs']}')
        for batch in batch_iter:
            img, label = batch['img'].to(config['device']), batch['label'].to(config['device'])
            img = img.to(config['device'], dtype=torch.float32, memory_format=torch.channels_last)
            label = label.to(config['device'], dtype = torch.long)

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

        scheduler.step()
        loss, acc = run_validation(model, valid_loader, criterion, config['device'])
        print(f"Epoch {epoch + 1}/{config['n_epochs']}, loss: {epoch_loss:.3f}, accuracy: {epoch_accuracy:.3f}")

    loss, acc = run_validation(model, valid_loader, criterion, config['device'])
    print(f"Test loss: {loss:.3f}, accuracy: {acc:.3f}")

    if config['save']:
        if config['platform'] == "kaggle":
            model_filename = get_weight_file_path_kaggle("/kaggle/working/papers_implement/CNN/DETR", config['save_name'], f'{epoch:02d}')
        else:
            model_filename = get_weight_file_path(config['save_name'], f'{epoch:02d}')
        
        print(f'Saving model {model_filename}')

        if not os.path.exists(config['output_dir']):
            os.makedirs(config['output_dir'])
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="ResNet50")
    parser.add_argument("--n_classes", type=int, default=20)
    parser.add_argument("--n_queries", type=int, default=100)
    parser.add_argument("--detr_pretrained", default=None)
    parser.add_argument("--preload", default=False)
    parser.add_argument("--preload_name", default="detr.pth")
    parser.add_argument("--platform", default="pc")
    parser.add_argument("--save", default=False)
    parser.add_argument("--save_name", default="detr.pth")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    config = get_config()
    config.update(vars(args))
    
    # Create output directory if it doesn't exist
    os.makedirs(config['output_dir'], exist_ok=True)
    if config['platform'] == "kaggle":
        config['TRAIN_DIR'] = Path('/kaggle/working/papers_implement/CNN/DETR/data/train')
        config['TEST_DIR'] = Path('/kaggle/working/papers_implement/CNN/DETR/data/test')
    
    train_model(config)