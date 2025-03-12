import torch.nn as nn

class PredictionFFN(nn.Module):
    def __init__(self, d_model, n_bbox, n_classes):
        super().__init__()
        self.bbox_linear = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, n_bbox) # centered normalized coordinate (x, y) and (w, h)
        )

        self.cls_linear = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, n_classes + 1) # 0 for background
        )
        
    def forward(self, x):
        bbox_pred = self.bbox_linear(x)
        cls_pred = self.cls_linear(x)
        return bbox_pred, cls_pred
        
