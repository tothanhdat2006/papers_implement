import torch
import torch.nn as nn
import math

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
    def __init__(self, dropout=0.1):
        super().__init__()
        self.norm_layer = AddNormLayer(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm_layer(x)))


class EncoderLayer(nn.Module):
    def __init__(self, self_attn_block, positionwise_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn_block = self_attn_block
        self.positionwise_feedforward = positionwise_feedforward

        self.residual_connection_layer = nn.Module([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, enc_input, attn_masks=None):
        enc_output = self.residual_connection_layer[0](enc_input, \
                                                    lambda enc_input: self.self_attn_block(enc_input, enc_input, enc_input, attn_masks))
        enc_output = self.residual_connection_layer[1](enc_input, self.positionwise_feedforward)
        return enc_output #QKV and attn_scores


class DecoderLayer(nn.Module):
    def __init__(self, self_attn_block, cross_attn_block, positionwise_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn_block = self_attn_block
        self.cross_attn_block = cross_attn_block
        self.positionwise_feedforward = positionwise_feedforward
        self.residual_connection_layer = nn.Module([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, dec_input, enc_output, src_mask, target_mask=None):
        dec_output = self.residual_connection_layer[0](dec_input, \
                                                    lambda dec_input: self.self_attn_block(dec_input, dec_input, dec_input, target_mask))
        dec_output = self.residual_connection_layer[1](dec_output, \
                                                    lambda dec_output: self.cross_attn_block(dec_output, enc_output, enc_output, src_mask))
        dec_output = self.residual_connection_layer[2](dec_output, self.positionwise_feedforward)
        return dec_output #QKV and attn_scores
        

        