import math

import torch
import torch.nn as nn

from transformer.sublayers import *

class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        return self.embedding(x) * math.sqrt(self.d_model) 


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout=0.1):
        super().__init__()
        self.norm_layer = NormLayer(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm_layer(x))) # Section 5.4: Residual dropout


class EncoderLayer(nn.Module):
    def __init__(self, features, self_attn_block, positionwise_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn_block = self_attn_block
        self.positionwise_feedforward = positionwise_feedforward

        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask=None):
        x = self.residual_connections[0](x, lambda x: self.self_attn_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.positionwise_feedforward)
        return x #QKV and attn_scores


class DecoderLayer(nn.Module):
    def __init__(self, features, self_attn_block, cross_attn_block, positionwise_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn_block = self_attn_block
        self.cross_attn_block = cross_attn_block
        self.positionwise_feedforward = positionwise_feedforward
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attn_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attn_block(x, enc_output, enc_output, src_mask))
        x = self.residual_connections[2](x, self.positionwise_feedforward)
        return x #QKV and attn_scores
        

        