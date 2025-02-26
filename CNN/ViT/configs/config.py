class Config:
    patches_sz = 10
    lr = 1e-3 # Table 3 - ViT-B/{16, 32} for imagenet-21k
    batch_size = 512
    weight_decay = 0.03
    # BERT 2019 - ViT-Base config
    n_layers = 12
    hid_size = 768
    mlp_size = 3072
    n_heads = 12
