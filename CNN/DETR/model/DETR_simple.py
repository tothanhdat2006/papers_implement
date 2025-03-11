import torch
import torch.nn as nn
from torchvision.models import resnet50
class DETR(nn.Module):
    def __init__(self, n_classes, hidden_dim, n_heads,
                 n_enc_layers, n_dec_layers, dim_feedforward):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, 1) # Projection layer
        self.transformer = nn.Transformer(hidden_dim, n_heads, n_enc_layers, n_dec_layers)

        self.linear_cls = nn.Linear(hidden_dim, n_classes+1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        self.query_pos_encoder = nn.Parameter(torch.randn(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.randn(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.randn(50, hidden_dim // 2))

    def forward(self, x):
        f = self.backbone(x)
        z0 = self.conv(f)
        H, W = z0.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        z0 = self.transformer(pos + z0.flatten(2).permute(2, 0, 1),
                              self.query_pos_encoder.unsqueeze(1))
        return self.linear_cls(z0), self.linear_bbox(z0).sigmoid()

