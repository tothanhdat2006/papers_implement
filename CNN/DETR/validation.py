import torch
from tqdm import tqdm
import torch.nn as nn

def run_validation(model, valid_loader, criterion, device):
    model.eval()
    valid_loss = 0.0
    valid_acc = 0.0

    with torch.no_grad():
        for batch in valid_loader:
            img, label = batch['img'].to(device), batch['label'].to(device)

            val_output = model(img)
            loss = criterion(val_output, label)
            acc = (val_output.argmax(dim=1) == label).sum().item() / label.size(0)

            valid_loss += loss / len(valid_loader)
            valid_acc += acc / len(valid_loader)

    model.train()
    return valid_loss, valid_acc