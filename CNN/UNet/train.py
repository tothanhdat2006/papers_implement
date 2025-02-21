import logging
from tqdm import tqdm
import wandb
import argparse

import os
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

from evaluate import evaluate
from unet.unet import UNET
from utils.load_data import ImageCustomDataset, MaskCustomDataset
from utils.dice_score import dice_loss

dir_images = Path('./data/imgs')
dir_masks = Path('./data/masks')
dir_checkpoint = Path('./checkpoints/')

def train_model(
        model, 
        device, 
        n_epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.,
        save_checkpoint: bool = True,
        img_scale: float = 0.3,
        amp: bool = True,
        weight_decay: float = 1e-8,
        momentum: float = 0.99,
):
    # 1. Create dataset
    dataset = MaskCustomDataset(dir_images, dir_masks, img_scale)

    # tmp_img = dataset[0]['image'].permute(1, 2, 0)
    # plt.imshow(np.asarray(tmp_img))
    # plt.show()

    # 2. Split into train / validation partitions
    n_val = int(val_percent * len(dataset))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator = torch.Generator().manual_seed(86))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_dataloader = DataLoader(train_set, shuffle=True, **loader_args)
    val_dataloader = DataLoader(val_set, shuffle=True, **loader_args)

    experiment = wandb.init(project='UNet', resume='allow')
    experiment.config.update(
        dict(n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {n_epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')
    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.GradScaler(enabled=amp)
    global_step = 0

    # 5. Begin training
    for epoch in range(1, n_epochs + 1):
        torch.cuda.empty_cache()
        model.train() # Set model to training mode
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{n_epochs}', unit='img') as pbar:
          for batch in train_dataloader:
              images, masks_true = batch['image'], batch['mask']
              images = images.to(device, dtype=torch.float32, memory_format=torch.channels_last)
              masks_true = masks_true.to(device, dtype = torch.long)
              
              with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                  masks_pred = model(images)
                  if model.n_classes == 1:
                      loss = criterion(masks_pred.squeeze(1), masks_true.float())
                      loss += dice_loss(nn.functional.sigmoid(masks_pred.squeeze(1)), masks_true.float(), multiclass=False)
                  else:
                      loss = criterion(masks_pred, masks_true)
                      loss += dice_loss(
                          nn.functional.softmax(masks_pred, dim = 1).float(), 
                          nn.functional.one_hot(masks_true, model.n_classes).permute(0, 3, 1, 2).float(), 
                          multiclass=True
                      )

              optimizer.zero_grad()
              grad_scaler.scale(loss).backward()
              grad_scaler.unscale_(optimizer)
              nn.utils.clip_grad_norm_(model.parameters(), 1.0)
              grad_scaler.step(optimizer)
              grad_scaler.update()

              pbar.update(images.shape[0])
              global_step += 1
              epoch_loss += loss.item()
              experiment.log({
                  'train loss': loss.item(),
                  'step': global_step,
                  'epoch': epoch
              })
              pbar.set_postfix(**{'loss (batch)': loss.item()})

              division_step = (n_train // (5 * batch_size))
              if division_step > 0:
                  if global_step % division_step == 0:
                      histograms = {}
                      for tag, value in model.named_parameters():
                          tag = tag.replace('/', '.')
                          if not (torch.isinf(value) | torch.isnan(value)).any():
                              histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                          if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                              histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                      val_score = evaluate(model, val_dataloader, device, amp)
                      scheduler.step(val_score)

                      logging.info('Validation Dice score: {}'.format(val_score))
                      try:
                          experiment.log({
                              'learning rate': optimizer.param_groups[0]['lr'],
                              'validation Dice': val_score,
                              'images': wandb.Image(images[0].cpu()),
                              'masks': {
                                  'true': wandb.Image(masks_true[0].float().cpu()),
                                  'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                              },
                              'step': global_step,
                              'epoch': epoch,
                              **histograms
                          })
                      except:
                          pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')            


def get_args():
    parser = argparse.ArgumentParser(description='Train UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.3, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == "__main__":
    print("Training...")
    args = get_args()
    
    model = UNET(n_channels=3, n_classes=args.classes)
    model = model.to(memory_format = torch.channels_last)
    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    model.to(device=device)
    try:
        train_model(
            model,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    
