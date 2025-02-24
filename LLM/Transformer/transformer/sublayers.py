import torch
import torch.nn as nn
import math

class AddNormLayer(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.w * (x - mean) / (std + self.eps) + self.b


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k # d_model // n_head
        self.d_v = d_v # d_model // n_head

        self.w_q = nn.Linear(d_model, n_head * d_k, bias = False)
        self.w_k = nn.Linear(d_model, n_head * d_k, bias = False)
        self.w_v = nn.Linear(d_model, n_head * d_v, bias = False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias = False)

        self.layer_norm = nn.LayerNorm(d_model, eps = 1e-6)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def ScaledDotProductAttention(self, q, k, v, mask=None, temperature=1):
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k) #Q @ K^T / sqrt(d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask==0, -1e9)

        # attn_scores = (batch, h, seq_len) -> (batch, h, seq_len, seq_len)
        attn_scores = self.dropout(torch.softmax(attn_scores, dim=-1)) # softmax(Q @ K^T / sqrt(d_k))
        # output = (batch, h, seq_len, seq_len) -> (batch, h, seq_len, d_v)
        output = torch.matmul(attn_scores, v) # softmax(Q @ K^T / sqrt(d_k)) @ V

        return output, attn_scores

    def forward(self, q, k, v, mask=None):
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        # Mask to 
        residual = q
 
        q = self.w_q(q).view(sz_b, len_q, self.n_head, self.d_k) # (batch, seq_len, d_model) -> (batch, len_q, n_head, d_k)
        k = self.w_k(k).view(sz_b, len_k, self.n_head, self.d_k) # (batch, seq_len, d_model) -> (batch, len_k, n_head, d_k)
        v = self.w_v(v).view(sz_b, len_v, self.n_head, self.d_k) # (batch, seq_len, d_model) -> (batch, len_v, n_head, d_k)

        # (batch, len_q, n_head, d_?) -> (batch, n_head, len_q, d_?)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        x, attn_scores = MultiHeadAttention.ScaledDotProductAttention(q, k, v, mask=mask)

        # (batch, h, seq_len, d_v) -> (batch, seq_len, h, d_v) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contigous().view(sz_b, len_q, -1)
        x = self.dropout(self.fc(x))
        x += residual

        return self.layer_norm(x), attn_scores


        
        




class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.Wb1 = nn.Linear(d_model, d_ff)
        self.Wb2 = nn.Linear(d_ff, d_model)

        self.layer_norm = nn.LayerNorm(d_model, eps = 1e-6)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        residual = x

        x = self.Wb2(torch.relu(self.Wb1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x
