import argparse
import wandb
import logging
from tqdm import tqdm 
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.load_data import load_data
from evaluate import evaluate
from resnet.resnet import ResNet, ResidualBlock

dir_checkpoint = Path('./checkpoints/')

def train_model(
    model,
    device,
    n_epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    val_percent: float = 0.2,
    save_checkpoint: bool = True,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
):
    '''
    Resize: shorter side randomly sampled in [256, 480]
    Crop: 224 x 224 crop is randomly sampled
    Flip: Horizontal flip
    Standardize: per-pixel mean subtracted, standardize color augmentation

    Initialize:
    mini-batch size: 256
    lr: 0.1, /10 when error plateaus
    iterations: 60 * 10^4 it
    weight decay: 1e-4
    momentum: 0.9
    dropout: no
    '''
    
    # 1. Load data
    classes_dict, train_dataloader, val_dataloader = load_data(dir_data='data', batch_size=batch_size, num_workers=0, val_percent=val_percent)
    
    experiment = wandb.init(project='ResNet', resume='allow')
    experiment.config.update(
        dict(n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint)
    )

    logging.info(f'''Starting training:
        Epochs:          {n_epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_dataloader)}
        Validation size: {len(val_dataloader)}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
    ''')

    # 2. Set up optimizer, loss function, learning rate scheduler
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum) 
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    global_step = 0

    # 3. Begin training
    train_losses = []
    val_losses = []
    n_train = len(train_dataloader)
    division_step = (n_train // (5 * batch_size))
    best_val_loss = float('inf')
    for epoch in range(1, n_epochs+1):
        epoch_loss = 0.0
        model.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{n_epochs}', unit='img') as pbar:
            for batch in train_dataloader:
                image, label = batch['image'], batch['label']
                image = image.to(device, dtype=torch.float32, memory_format=torch.channels_last)
                label = label.to(device, dtype=torch.long)

                optimizer.zero_grad()
                label_pred = model(image)
                loss = criterion(label_pred, label)
                loss.backward()
                optimizer.step()
                
                pbar.update(image.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_loss, val_acc = evaluate(model, val_dataloader, criterion, device)
                        scheduler.step(val_loss)
                        val_losses.append(val_loss)

                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_loss,
                                'images': wandb.Image(image[0].cpu()),
                                'masks': {
                                    'true': label,
                                    'pred': label_pred.argmax(dim=1)[0].float().cpu(),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        train_losses.append(epoch_loss / n_train)
        print(f"Epoch {epoch}: Training Loss = {train_losses[-1]}, Validation Loss = {val_losses[-1]}")
        if val_losses[-1] < best_val_loss:  # Check for improvement
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)  # Ensure dir exists
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / f'checkpoint_ResNet_epoch{epoch}.pth'))
            logging.info(f'Checkpoint saved! Validation loss improved from {best_val_loss:.4f} to {val_losses[-1]:.4f}')
            best_val_loss = val_losses[-1]


def get_args():
    parser = argparse.ArgumentParser(description='Train ResNet')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', dest='lr', metavar='LR', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from .pth file')
    parser.add_argument('--classes', '-c', dest='classes', metavar='C', type=int, default=75, help='Number of classes')
    return parser.parse_args()


if __name__ == "__main__":
    print('Training...')
    args = get_args()

    n_blocks_list = [2, 2, 2, 2]
    model = ResNet(residual_block=ResidualBlock, 
                   n_blocks_list=n_blocks_list,
                   n_classes=args.classes)
    model = model.to(memory_format = torch.channels_last)
    logging.info(f'Model trained on {args.classes} classes\n'
                 f'Number of blocks per conv:\n'
                 f'\t{n_blocks_list[0]} for conv2_x\n'
                 f'\t{n_blocks_list[1]} for conv3_x\n'
                 f'\t{n_blocks_list[2]} for conv4_x\n'
                 f'\t{n_blocks_list[3]} for conv5_x\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    print(f'Using device: {device}')
    model.to(device=device)
    try:
        train_model(model,
                    device,
                    n_epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    val_percent=args.val / 100)
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training.')
        torch.cuda.empty_cache()
        print("Not enough memory to train model")
    
