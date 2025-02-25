class Config:
    batch_size = 8,
    num_epochs = 20,
    lr = 1e-4,
    seq_len = 350,
    d_model = 512,
    lang_src = "en",
    lang_tgt = "it",
    model_folder = "weights",
    model_file = "tmodel_",
    preload = None,
    tokenizer_file = "tokenizer_{0}.json",
    experiment_name = "runs/tmodel"
