import math

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model) # (seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        self.W_q = nn.Linear(d_model, n_heads * self.d_k)
        self.W_k = nn.Linear(d_model, n_heads * self.d_k)
        self.W_v = nn.Linear(d_model, n_heads * self.d_v)
        self.W_o = nn.Linear(n_heads * self.d_v, d_model)

    @staticmethod
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        score = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        score = torch.softmax(score, dim=-1)
        score = score @ v
        return score
        
    def forward(self, X, mask=None):
        



class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout

        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.FFN = FFN(d_model, d_model)
        

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()



class Transformer(nn.Module):
    def __init__(self, d_model, n_heads=8, n_enc_layers=6, n_dec_layers=6, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.dropout = dropout

    def forward(self, X, fixed_pos_enc):
        
        return X
        
        
