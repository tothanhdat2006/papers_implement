import math
import torch
import torch.nn as nn

from .backbone import build_backbone
from .transformer import build_transformer
from .FFN import PredictionFFN
    
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

class DETR(nn.Module):
    '''
    DETR model
    Args:
        backbone: backbone model
        transformer: transformer model
        n_classes: number of classes
        n_queries: number of objectqueries (N in the paper)
        aux_loss: whether to use auxiliary loss
    
        
    '''
    def __init__(self, backbone, transformer, n_classes, n_queries=100, in_channels=3, d_model=256):
        super().__init__()
        self.backbone = backbone
        self.conv1x1 = nn.Conv2d(2048, d_model, kernel_size=1)
        
        self.d_model = d_model
        self.query_embed = nn.Embedding(n_queries, d_model) # (n_queries, d_model)
        self.transformer = transformer
        self.prediction_ffn = PredictionFFN(d_model=d_model, n_bbox=n_queries, n_classes=n_classes)

    def forward(self, x):
        # x = (batch_size, 3, H0, W0)
        f = self.backbone(x) # (batch_size, 2048, H, W)
        z0 = self.conv1x1(f) # (batch_size, d_model, H, W)

        B, C, H, W = z0.shape # seq_len = HW
        z0 = z0.view(-1, B, C) # (HW, batch_size, d_model)
        pos_embed = PositionalEncoding(self.d_model, H * W, dropout=0.1) # (HW, batch_size, d_model) 
        pos_embed_z0 = pos_embed(z0)
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1) # (batch_size, n_queries, d_model)

        transformer_out = self.transformer(pos_embed_z0, query_embed) # (batch_size, n_queries, d_model)
        transformer_out = transformer_out.transpose(0, 1) # (n_queries, batch_size, d_model)

        bbox_pred, cls_logits = self.prediction_ffn(transformer_out) # (batch_size, n_queries, n_bbox), (batch_size, n_queries, n_classes + 1)
        cls_pred = torch.softmax(cls_logits, dim=2)
        return bbox_pred, cls_pred



def build_DETR(config):
    backbone = build_backbone(config['backbone'], config['n_classes'])
    transformer = build_transformer(config)
    model = DETR(backbone, transformer, config['n_classes'], config['n_queries'])

    if config['detr_pretrained'] is not None and config['detr_pretrained'] != 'None':
        print(f"Loading pretrained model from {config['detr_pretrained']}")
        ckpt = torch.load(config['detr_pretrained'], map_location=config['device'], weights_only=True)
        model.load_state_dict(ckpt["model"])
    else:
        print("Training from scratch - no pretrained model loaded")
    return model
