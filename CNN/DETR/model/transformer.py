import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_seq_len(x):
    if x.ndim == 2:
        # (seq_len, d_model)
        return x.shape[0]
    elif x.ndim == 3:
        # (batch_size, seq_len, d_model)
        return x.shape[1]
    else:
        raise ValueError(f"Input tensor has {x.ndim} dimensions. Expected 2 or 3.")


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


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout

        self.proj = nn.Linear(d_model, 3 * d_model)

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        self.in_proj_w = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.in_proj_b = nn.Parameter(torch.empty(3 * d_model))

        self.out_proj = nn.Linear(d_model, d_model)

    def _scaled_dot_product_attention(self, q, k, v, dropout=0.0):
        score = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        score = F.softmax(score, dim=-1)
        score = score @ v
        if dropout > 0.0:
            score = F.dropout(score, p=dropout)
        return score
    
    def _proj(self, q, k, v):
        d_model = q.shape[-1]
        w = self.in_proj_w        
        b = self.in_proj_b

        if q is k:
            # Self-attention
            q, k, v = F.linear(q, w, b).chunk(3, dim=-1)
            return [q, k, v]
        else:
            # Cross-attention
            w_q, w_kv = w.split([d_model, 2 * d_model])
            if b is None:
                b_q, b_kv = None
            else:
                b_q, b_kv = b.split([d_model, 2 * d_model])
            
            q = F.linear(q, w_q, b_q)
            k, v = F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
            return [q, k, v]

    def _out_proj(self, attn_out):
        return F.linear(attn_out, self.out_proj.weight, self.out_proj.bias)

    def forward(self, q, k, v):
        '''
        Args:
            q = (n_queries, batch_size, d_model)
            k = (n_keys, batch_size, d_model)
            v = (n_keys, batch_size, d_model)
        
        Returns:
            out = (n_queries, batch_size, d_model)
        '''
        n_queries, batch_size, d_model = q.shape

        q, k, v = self._proj(q, k, v)
        q = q.contiguous().view(n_queries, batch_size * self.n_heads, self.d_k).transpose(0, 1)
        k = k.contiguous().view(k.shape[0], batch_size * self.n_heads, self.d_k).transpose(0, 1)
        v = v.contiguous().view(v.shape[0], batch_size * self.n_heads, self.d_v).transpose(0, 1)

        attn_out = self._scaled_dot_product_attention(q, k, v, dropout=self.dropout)
        attn_out = attn_out.transpose(0, 1).contiguous().view(n_queries * batch_size, d_model)
        out = self._out_proj(attn_out)
        out = out.view(n_queries, batch_size, d_model)
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
    def __init__(self, d_model, d_ff, n_heads, dropout):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.FFN = FFN(d_model, d_ff, dropout)
        self.norm1 = ResidualConnection(d_model, dropout)
        self.norm2 = ResidualConnection(d_model, dropout)

    def forward(self, x):
        x = self.norm1(x, lambda x: self.attn(x, x, x))
        x = self.norm2(x, self.FFN)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.FFN = FFN(d_model, d_ff, dropout)
        self.norm1 = ResidualConnection(d_model, dropout)
        self.norm2 = ResidualConnection(d_model, dropout)
        self.norm3 = ResidualConnection(d_model, dropout)

    def forward(self, object_queries, enc_output):
        object_queries = self.norm1(object_queries, lambda x: self.self_attn(x, x, x))
        object_queries = self.norm2(object_queries, lambda x: self.cross_attn(x, enc_output, enc_output))
        object_queries = self.norm3(object_queries, self.FFN)
        return object_queries
    

class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_enc_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, n_heads, dropout) 
                                     for _ in range(n_enc_layers)])
        self.norm = NormLayer(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    

class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_dec_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, n_heads, dropout) 
                                     for _ in range(n_dec_layers)])
        self.norm = NormLayer(d_model)

    def forward(self, object_queries, enc_output):
        for layer in self.layers:
            object_queries = layer(object_queries, enc_output)
        return self.norm(object_queries)
    

class Transformer(nn.Module):
    def __init__(self, d_model=256, d_ff=2048,n_heads=8, n_enc_layers=6, n_dec_layers=6, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(d_model, d_ff, n_heads, n_enc_layers, dropout)
        self.decoder = Decoder(d_model, d_ff, n_heads, n_dec_layers, dropout)

    def forward(self, x, obj_queries):
        '''
        x = (HW, batch_size, d_model)
        obj_queries = (n_queries, batch_size, d_model)
        '''
        enc_output = self.encoder(x)
        dec_output = self.decoder(obj_queries, enc_output)
        return dec_output
        
        
def build_transformer(config: dict):
    transformer = Transformer(d_model=config['d_model'], d_ff=config['d_ff'], n_heads=config['n_heads'], n_enc_layers=config['n_enc_layers'], n_dec_layers=config['n_dec_layers'], dropout=config['dropout'])
    return transformer
