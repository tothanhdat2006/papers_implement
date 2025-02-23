import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super().__init__()



class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super().__init__()