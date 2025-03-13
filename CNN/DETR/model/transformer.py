import math

import torch
import torch.nn as nn

def get_seq_len(x):
    if x.ndim == 2:
        # (seq_len, d_model)
        return x.shape[0]
    elif x.ndim == 3:
        # (batch_size, seq_len, d_model)
        return x.shape[1]
    else:
        raise ValueError(f"Input tensor has {x.ndim} dimensions. Expected 2 or 3.")

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


class NormLayer(nn.Module):
    def __init__(self, d_model: int, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.w * (x - mean) / (std + self.eps) + self.b
    

class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout=0.1):
        super().__init__()
        self.norm_layer = NormLayer(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm_layer(x))) # Section 5.4: Residual dropout


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

        self.W_q = nn.Linear(d_model, self.d_k)
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, self.d_v)
        self.W_o = nn.Linear(n_heads * self.d_v, d_model)

    @staticmethod
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        score = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        score = torch.softmax(score, dim=-1)
        score = score @ v
        return score
    
    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        # => (batch_size, n_heads, seq_len, d_k)

        out = self.scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        out = self.W_o(out)
        return out
                         

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.FFN = FFN(d_model, d_model, dropout)
        self.norm1 = ResidualConnection(d_model, dropout)
        self.norm2 = ResidualConnection(d_model, dropout)

    def forward(self, x, mask=None):
        seq_len = get_seq_len(x)
        x = self.pe(self.d_model, seq_len, self.dropout)(x)
        x = self.norm1(x, self.self_attn(x, x, x, mask))
        x = self.norm2(x, self.FFN(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout

        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = ResidualConnection(d_model, dropout)
        self.cross_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = ResidualConnection(d_model, dropout)
        self.FFN = FFN(d_model, d_model)
        self.norm3 = ResidualConnection(d_model, dropout)

    def forward(self, x, object_queries, enc_output, src_mask, tgt_mask):
        x = x + object_queries
        x = self.norm1(x, self.self_attn(x, x, x, tgt_mask))
        x = self.norm2(x, self.cross_attn(x, enc_output, enc_output, src_mask))
        x = self.norm3(x, self.FFN(x))
        return x
    

class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, n_enc_layers, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.dropout = dropout

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, dropout) for _ in range(n_enc_layers)]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
    

class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, n_dec_layers, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_dec_layers = n_dec_layers
        self.dropout = dropout

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, dropout) for _ in range(n_dec_layers)]
        )

    def forward(self, x, object_queries, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, object_queries, enc_output, src_mask, tgt_mask)
        return x
    

class Transformer(nn.Module):
    def __init__(self, d_model, n_heads=8, n_enc_layers=6, n_dec_layers=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.dropout = dropout

        self.encoder = Encoder(d_model, n_heads, n_enc_layers, dropout)
        self.decoder = Decoder(d_model, n_heads, n_dec_layers, dropout)

    def forward(self, x, obj_queries):
        enc_output = self.encoder(x)
        dec_output = self.decoder(x, enc_output, obj_queries)
        return dec_output
        
        
