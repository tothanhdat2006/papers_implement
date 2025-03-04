import torch
import torch.nn as nn

from transformer.sublayers import *
from transformer.layers import *

# Section 3.5: Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Matrix (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Vector (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  

        # Simplified version of the original formula
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Matrix (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # Don't train the positional encoding
        return self.dropout(x) # Section 5.4: Residual dropout


class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm_layer = NormLayer(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm_layer(x)


class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm_layer = NormLayer(features)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm_layer(x)


class ProjectionLayer(nn.Module):  
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: InputEmbedding, tgt_embed: InputEmbedding,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 proj_layer: ProjectionLayer):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj_layer = proj_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    # Ensure params order
    def decode(self, enc_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, enc_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.proj_layer(x)
    

def build_transformer(src_vocab_sz, tgt_vocab_sz, src_seq_len, tgt_seq_len,
                      d_model=512, d_ff=2048, d_k=64, d_v=64, n_head=8, n_layers=6, dropout=0.1):
    # Build the model
    src_embed = InputEmbedding(d_model, src_vocab_sz)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_sz)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(n_layers):
        encoder_self_attn_block = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout)
        encoder_positionwise_feedforward = PositionwiseFeedForward(d_model, d_ff, dropout)
        encoder_blocks.append(EncoderLayer(d_model, encoder_self_attn_block, encoder_positionwise_feedforward, dropout))

    decoder_blocks = []
    for _ in range(n_layers):
        decoder_self_attn_block = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout)
        decoder_cross_attn_block = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout)
        decoder_positionwise_feedforward = PositionwiseFeedForward(d_model, d_ff, dropout)
        decoder_blocks.append(DecoderLayer(d_model, decoder_self_attn_block, decoder_cross_attn_block, decoder_positionwise_feedforward, dropout))

    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    proj_layer = ProjectionLayer(d_model, tgt_vocab_sz)
    
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, proj_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
