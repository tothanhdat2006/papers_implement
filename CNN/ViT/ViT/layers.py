import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, ch, img_sz, patchs_sz, emb_dim):
        super().__init__()
        self.patchs_sz = patchs_sz
        self.proj = nn.Sequential(
            
        )

class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches, emb_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))

    def forward(self, x):
        return torch.concat([x, self.embedding.repeat(x.size(0), 1, 1)], dim=1)

class ResidualConnection(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.norm_layer = nn.LayerNorm()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm_layer(x)))

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
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