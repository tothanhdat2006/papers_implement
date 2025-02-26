import math

import torch
import torch.nn as nn

class AddNormLayer(nn.Module):
    def __init__(self, features: int, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.w * (x - mean) / (std + self.eps) + self.b


# Section 3.2.2: Multi Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k # d_model // n_head
        self.d_v = d_v # d_model // n_head

        self.w_q = nn.Linear(d_model, n_head * d_k, bias = False)
        self.w_k = nn.Linear(d_model, n_head * d_k, bias = False)
        self.w_v = nn.Linear(d_model, n_head * d_v, bias = False)
        self.w_o = nn.Linear(n_head * d_v, d_model, bias = False)

        self.layer_norm = nn.LayerNorm(d_model, eps = 1e-6)
        self.dropout = nn.Dropout(dropout)

    # Section 3.2.1: Scaled Dot-Product Attention
    @staticmethod
    def ScaledDotProductAttention(self, q, k, v, mask=None):
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k) #Q @ K^T / sqrt(d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill_(mask==0, -1e9)

        # attn_scores = (batch, h, seq_len) -> (batch, h, seq_len, seq_len)
        attn_scores = self.dropout(torch.softmax(attn_scores, dim=-1)) # softmax(Q @ K^T / sqrt(d_k))
        # output = (batch, h, seq_len, seq_len) -> (batch, h, seq_len, d_v)
        output = attn_scores @ v # softmax(Q @ K^T / sqrt(d_k)) @ V

        return output, attn_scores

    def forward(self, q, k, v, mask=None):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = q.view(q.shape[0], q.shape[1], self.n_head, self.d_k) # (batch, seq_len, d_model) -> (batch, len_q, n_head, d_k)
        k = k.view(k.shape[0], k.shape[1], self.n_head, self.d_k) # (batch, seq_len, d_model) -> (batch, len_k, n_head, d_k)
        v = v.view(v.shape[0], v.shape[1], self.n_head, self.d_k) # (batch, seq_len, d_model) -> (batch, len_v, n_head, d_k)

        # (batch, len_q, n_head, d_k) -> (batch, n_head, len_q, d_k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            if mask.dim() == 3: # Important
                mask = mask.unsqueeze(1)

        x, self.attn_scores = MultiHeadAttention.ScaledDotProductAttention(self, q, k, v, mask=mask)

        # Concat all heads together
        # (batch, h, seq_len, d_v) -> (batch, seq_len, h, d_v) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_head * self.d_v)

        return self.w_o(x)


# Section 3.3: Position-wise Feed-Foward Networks
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.Wb1 = nn.Linear(d_model, d_ff)
        self.Wb2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.Wb2(self.dropout(torch.relu(self.Wb1(x))))
