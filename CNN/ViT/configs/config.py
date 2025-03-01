class Config:
    # Training
    platform = None
    train_size = 100
    test_size = 10
    n_epochs = 10
    batch_size = 64
    lr = 1e-3 # Table 3 - ViT-B/{16, 32} for imagenet-21k
    weight_decay = 0.03
    gamma = 0.7

    # Images
    n_classes = 1000
    patches_sz = 10
    img_sz = 256
    ch = 3
    
    # BERT 2019 - ViT-Base config
    n_layers=12
    n_heads=12
    hid_dim=768
    mlp_dim=3072
    pool='cls'
