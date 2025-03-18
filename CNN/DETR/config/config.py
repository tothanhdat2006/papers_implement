import torch
from pathlib import Path

def get_config():
    config = {
        # Transformer
        'd_model': 256,
        'd_ff': 2048,
        'n_heads': 8,
        'n_enc_layers': 6,
        'n_dec_layers': 6,
        'dropout': 0.1,

        # Backbone
        'backbone': 'ResNet101',
        'n_classes': 1000,  # This will be updated based on dataset
        'n_queries': 100,

        # DETR
        'aux_loss': True,
        'detr_pretrained': None,  # Path to pretrained DETR model, None to train from scratch

        # Data
        'TRAIN_DIR': Path('data/train'),
        'TEST_DIR': Path('data/test'),
        'train_size': 100,
        'test_size': 10,
        'num_workers': 4,
        'pin_memory': True,
        'shuffle': True,

        # Training
        'n_epochs': 5,
        'batch_size': 16,  # Batch size for training and validation
        'platform': 'pc',
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'gamma': 0.7,  # Learning rate decay
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        # Saving and Loading
        'output_dir': 'output',
        'save': True,
        'save_name': 'detr.pth',
        'preload': False,
        'preload_name': 'detr.pth',
    }
    return config

def get_weight_file_path(model_basename: str, epoch: int):
    model_file_name = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_file_name)
    
def get_weight_file_path_kaggle(dir: str, model_basename: str, epoch: int):
    model_file_name = f"{model_basename}{epoch}.pt"
    return str(Path('.') / dir / model_file_name)
