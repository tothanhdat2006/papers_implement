import torch
import torch.nn as nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class FeedForward(nn.Module):
    def __init__(self, hid_dim=768, mlp_dim=3072, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hid_dim),
            nn.Linear(hid_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hid_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, n_heads=12, hid_dim=768, d_k=64, dropout=0.0):
        super().__init__()
        d_model = n_heads * d_k
        proj_out = not (n_heads == 1 and d_k == hid_dim)

        self.n_heads = n_heads
        self.scale = d_k ** -0.5

        self.norm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(hid_dim, d_model * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(d_model, hid_dim),
            nn.Dropout(dropout)
        ) if proj_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # (b_sz, seq_len, n_heads, d_k) -> (b_sz, n_heads, seq_len, d_k)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), qkv)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = nn.Softmax(dim=-1)(dots)
        attn = self.dropout(attn)
        attn = attn @ v

        out = rearrange(attn, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer_Encoder(nn.Module):
    def __init__(self, n_layers=12, n_heads=12, hid_dim=768, mlp_dim=3072, d_k=64, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hid_dim)
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                Attention(n_heads, hid_dim, d_k, dropout),
                FeedForward(hid_dim, mlp_dim, dropout)
            ]))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer[0](x) + x
            x = layer[1](x) + x
        
        return self.norm(x)

class VisionTransformer(nn.Module):
    def __init__(self, img_sz=(256, 256), patches_sz=(8, 8), num_classes=1000, n_layers=12, n_heads=12, hid_dim=768, mlp_dim=3072, pool='cls', ch=3, d_k=64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = img_sz
        patch_height, patch_width = patches_sz
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by patch size'
    
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = ch * patch_height * patch_width
         
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hid_dim),
            nn.LayerNorm(hid_dim)
        )

        self.pos_emb = nn.Parameter(torch.randn(1, num_patches + 1, hid_dim)) # 1D position embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, hid_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.tranformer_encoder = Transformer_Encoder(n_layers, n_heads, hid_dim, mlp_dim, d_k, dropout)
        
        assert pool in {'cls', 'mean'}, \
            'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(hid_dim, num_classes)

    def forward(self, x):
        # x = (B, C, H, W)
        # Patch -> Pos_embed
        xp = self.to_patch_embedding(x) # b (h w) (d_model c)
        b_sz, n_heads, _ = xp.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b_sz)
        xp = torch.cat((cls_tokens, xp), dim=1)
        xp += self.pos_emb[:, :(n_heads+1)]
        xp = self.dropout(xp)

        # xp = (N, P*P*C) <- N = (HW)/P**2
        # Project to D dimension
        xp = self.tranformer_encoder(xp)
        xp = xp.mean(dim=1) if self.pool == 'mean' else xp[:, 0] # x_class

        # MLP head -> classes
        xp = self.to_latent(xp)
        out = self.mlp_head(xp)
        return out
         

