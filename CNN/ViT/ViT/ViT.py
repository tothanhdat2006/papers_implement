import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, patches_sz, d_model, n_head):
        self.patches_sz = patches_sz
        self.d_model = d_model
        self.n_head = n_head

    def forward(self, x):
        # x = (H, W, C)
        # Patch -> Pos_embed
        x = self.patch(x)
        xp = self.pos_embed(x)
        # xp = (N, P*P*C) <- N = (HW)/P**2
        # Project to D dimension
        xp = self.lin_proj(xp)
        xp = self.encode(xp)
         

