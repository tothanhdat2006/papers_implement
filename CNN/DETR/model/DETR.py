import torch
import torch.nn as nn

from .backbone import build_backbone
from .transformer import PositionalEncoding, build_transformer
from .FFN import PredictionFFN
    
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
    def __init__(self, backbone, transformer, n_classes, n_queries, aux_loss=False):
        super().__init__()
        self.backbone = backbone
        self.conv1x1 = nn.Conv2d(backbone.n_channels, transformer.d_model, kernel_size=1)
        
        self.pe = PositionalEncoding(d_model=transformer.d_model, seq_len=transformer.seq_len)
        self.transformer = transformer

        self.prediction_ffn = PredictionFFN(d_model=transformer.d_model, n_bbox=n_queries, n_classes=n_classes)

    def forward(self, x, obj_queries):
        f = self.backbone(x) # (batch_size, 2048, H, W)
        z0 = self.conv1x1(f) # (batch_size, d_model, H, W)

        B, C, H, W = z0.shape
        x = x.view(B, C, -1) # (batch_size, d_model, HW)
        z0 = self.pe(z0) # (batch_size, d_model, HW)
        z0 = self.transformer(z0, obj_queries) # (batch_size, d_model, HW)
        
        bbox_pred, cls_logits = self.prediction_ffn(z0) # (batch_size, HW, n_bbox), (batch_size, HW, n_classes + 1)
        cls_pred = torch.softmax(cls_logits, dim=2)
        return bbox_pred, cls_pred


def build_DETR(args):
    backbone = build_backbone(args.backbone, args.n_classes)
    transformer = build_transformer()
    model = DETR(backbone, transformer, args.n_classes, args.n_queries, args.aux_loss)

    if args.detr_pretrained:
        ckpt = torch.load(args.detr_pretrained, map_location=args.device)
        model.load_state_dict(ckpt["model"])
    return model
