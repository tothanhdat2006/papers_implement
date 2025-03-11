import torch
import torch.nn as nn

from .backbone import build_backbone
from .transformer import PositionalEncoding, build_transformer


class DETR(nn.Module):
    def __init__(self, backbone, transformer, n_classes, n_queries, aux_loss=False):
        super().__init__()
        self.backbone = backbone
        self.conv1x1 = nn.Conv2d(backbone.n_channels, transformer.d_model, kernel_size=1)
        self.pe = PositionalEncoding(d_model=transformer.d_model, seq_len=transformer.seq_len)
        
        self.transformer = transformer

        self.linear_cls = nn.Linear(transformer.d_model, n_classes+1)
        self.linear_bbox = nn.Linear(transformer.d_model, 4)

    def forward(self, x):
        f = self.backbone(x) # (batch_size, 2048, H, W)
        z0 = self.conv1x1(f) # (batch_size, d_model, H, W)
        
        pos = self.pe(z0) # (batch_size, d_model, H, W)

        z0 = z0.flatten(2).permute(2, 0, 1) # (HW, batch_size, d_model)
        z0 = self.transformer(z0, pos) # (HW, batch_size, d_model)


def build_DETR(args):
    backbone = build_backbone(args.backbone, args.n_classes)
    transformer = build_transformer()
    model = DETR(backbone, transformer, args.n_classes, args.n_queries, args.aux_loss)

    if args.detr_pretrained:
        ckpt = torch.load(args.detr_pretrained, map_location=args.device)
        model.load_state_dict(ckpt["model"])
    return model
