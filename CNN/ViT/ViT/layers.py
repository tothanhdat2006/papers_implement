import torch
import torch.nn as nn

class ResidualConnection(nn.Module):
    def __init__(self):
        self.norm_layer = nn.LayerNorm()

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        self.d_model = d_model
        self.n_head = n_head
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.w_q = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_head * self.d_v, bias=False)
        self.w_o = nn.Linear(n_head * self.d_v, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def attention(self, q, k, v, mask):
        x = (q @ k.transpose(-2, -1)) / torch.sqrt(self.d_k)
        if mask is not None:
            x = x.masked_fill_(mask==0, -1e9)

        # (batch, h, seq_len) -> (batch, h, seq_len, seq_len) (to be modified)
        x = torch.softmax(x, dim=-1)
        return x @ v, x

    def forward(self, q, k, v, mask=None):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)