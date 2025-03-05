import torch
import torch.nn as nn

class RoIAlign(nn.Module):
    def __init__(self):
        super().__init__()