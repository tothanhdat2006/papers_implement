import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff

@torch.inference_mode()
def evaluate(model, dataloader, device, amp):
    model.eval() # Set to evaluation mode
    test_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            images, masks_true = batch['image'], batch['mask']
            images = images.to(device, dtype=torch.float32, memory_format=torch.channels_last)
            masks_true = masks_true.to(device, dtype = torch.long)
            masks_pred = model(images)

            if model.n_classes == 1:
                assert masks_true.min() >= 0 and masks_true.max() <= 1, 'True mask indices should be in [0, 1]'
                masks_pred = (F.sigmoid(masks_pred) > 0.5).float()
                # Dice score: (2 * |X| * |Y|) / (|X| + |Y|) 
                test_loss += dice_coeff(masks_pred, masks_true, reduce_batch_first=False)
            else:
                assert masks_true.min() >= 0 and masks_true.max() < model.n_classes, 'True mask indices should be in [0, n_classes)'
                masks_true = F.one_hot(masks_true, model.n_classes).permute(0, 3, 1, 2).float()
                masks_pred = F.one_hot(masks_pred.argmax(dim = 1), model.n_classes).permute(0, 3, 1, 2).float()
                # Dice score
                test_loss += multiclass_dice_coeff(masks_pred[:, 1:], masks_true[:, 1:], reduce_batch_first=False)

    test_loss = test_loss / max(len(dataloader), 1)

    model.train() # Set back to training mode
    return test_loss