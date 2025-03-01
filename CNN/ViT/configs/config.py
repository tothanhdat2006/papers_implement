from pathlib import Path

class Config:
    def __init__(self):
        # Training
        self.platform = None
        self.train_size = 100
        self.test_size = 10
        self.n_epochs = 10
        self.batch_size = 64
        self.lr = 1e-3 # Table 3 - ViT-B/{16, 32} for imagenet-21k
        self.weight_decay = 0.03
        self.gamma = 0.7

        # Images
        self.n_classes = 1000
        self.patches_sz = 10
        self.img_sz = 256
        self.ch = 3
        
        # BERT 2019 - ViT-Base config
        self.n_layers=12
        self.n_heads=12
        self.hid_dim=768
        self.mlp_dim=3072
        self.pool='cls'

        # Saving & Preloading path
        self.save = False
        self.save_name = None
        self.preload = None
        self.preload_name = None
        self.model_folder = "weights"

    def get_weight_file_path(self, model_basename: str, epoch: int):
        model_file_name = f"{model_basename}{epoch}.pt"
        return str(Path('.') / self.model_folder / model_file_name)
    
    def get_weight_file_path_kaggle(self, dir: str, model_basename: str, epoch: int):
        model_file_name = f"{model_basename}{epoch}.pt"
        return str(Path('.') / dir / self.model_folder / model_file_name)
