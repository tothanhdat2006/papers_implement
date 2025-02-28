import torch
import torch.nn as nn

from ViT.layers import *

class VisionTransformer(nn.Module):
    def __init__(self, ch=3, img_sz=224, patches_sz=8, n_layers=12, n_head=12, hid_size=768, mlp_size=3072, dropout=0.1):
        super().__init__()
        self.ch = ch
        self.img_size = img_sz
        self.patches_sz = patches_sz
        self.n_layers = n_layers
        self.n_head = n_head
        self.hid_size = hid_size
        self.mlp_size = mlp_size
        self.dropout = dropout

    def forward(self, x):
        # x = (H, W, C)
        # Patch -> Pos_embed
        x = self.patch(x)
        xp = self.pos_embed(x)
        # xp = (N, P*P*C) <- N = (HW)/P**2
        # Project to D dimension
        xp = self.lin_proj(xp)
        xp = self.encode(xp)
         

