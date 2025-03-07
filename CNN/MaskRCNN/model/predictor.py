from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, mid_channels, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, mid_channels)
        self.cls_score = nn.Linear(mid_channels, n_classes)
        self.bbox_pred = nn.Linear(mid_channels, n_classes * 4)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        score = self.cls_score(x)
        bbox_delta = self.bbox_pred(x)

        return score, bbox_delta
    
class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, layers, dim_reduced, n_classes):
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_feature in enumerate(layers, 1):
            d[f'mask_fcn{layer_idx}'] = nn.Conv2d(next_feature)
            d[f'relu{layer_idx}'] = nn.ReLU(inplace=True)
            next_feature = layer_feature

        d['mask_conv5'] = nn.ConvTranspose2d(next_feature, dim_reduced, kernel_size=2, stride=2, padding=0)
        d['relu5'] = nn.ReLU(inplace=True)
        d['mask_fcn_logits'] = nn.Conv2d(dim_reduced, n_classes, kernel_size=1, stride=1, padding=0) # 1 x 1 conv as linear func
        super().__init__(d)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')