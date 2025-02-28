from pathlib import Path

class Config:
    def __init__(self):
        super().__init__()
        self.train_size = 100
        self.batch_size = 8
        self.n_epochs = 25
        self.lr = 1e-4
        self.seq_len = 350
        self.d_model = 512
        self.d_ff = 2048
        self.d_k = 64
        self.d_v = 64
        self.n_head = 8
        self.n_layers = 6
        self.datasource = "opus_books"
        self.lang_src = "en"
        self.lang_tgt = "it"
        self.model_folder = "weights"
        self.model_basename = "tmodel_"
        self.preload = None
        self.tokenizer_file = "tokenizer_{0}.json"
        self.experiment_name = "runs/tmodel"
        self.platform = None

    def get_weight_file_path(self, epoch: int):
        model_file_name = f"{self.model_basename}{epoch}.pt"
        return str(Path('.') / self.model_folder / model_file_name)
    
    def get_weight_file_path_kaggle(self, dir: str, epoch: int):
        model_file_name = f"{self.model_basename}{epoch}.pt"
        return str(Path('.') / dir / self.model_folder / model_file_name)