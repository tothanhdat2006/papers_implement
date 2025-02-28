import os
import argparse
from pathlib import Path

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from configs.config import Config
from utils.preprocess import get_ds
from transformer.transformer import build_transformer 
from validation import run_validation
from translate import translate

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config.seq_len, config.seq_len, config.d_model)
    return model

def train_model(config, device):
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")

    if config.platform == "kaggle":
        Path("/kaggle/working/papers_implement/LLM/Transformer/" + config.model_folder).mkdir(parents=True, exist_ok=True)
    else:
        Path(config.model_folder).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config.experiment_name)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9) # Section 5.3: Optimizer
    initial_epoch = 0
    global_step = 0

    if config.preload:
        if config.platform == "kaggle":
            model_filename = config.get_weight_file_path_kaggle("/kaggle/working/papers_implement/LLM/Transformer", config.preload)
        else:
            model_filename = config.get_weight_file_path(config.preload)

        print(f'Preloading model {model_filename}')

        if torch.cuda.is_available():
            state = torch.load(model_filename, weights_only=True)
        else:
            state = torch.load(model_filename, map_location=torch.device("cpu"), weights_only=True)

        print("Preloading model complete!")
        print("Loading model state...")
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1).to(device) # Section 5.4: Label Smoothing

    for epoch in range(initial_epoch, config.n_epochs):
        batch_iter = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iter:
            model.train()
            encoder_input = batch['encoder_input'].to(device) # (batch_sz, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch_sz, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_sz, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch_sz, 1, seq_len, seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_sz, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch_sz, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch_sz, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) # (batch_sz, seq_len)

            # (batch_sz, seq_len, tgt_vocab_size) -> # (batch_sz * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iter.set_postfix({f"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config.seq_len, device,
                        print_msg=lambda msg: batch_iter.write(msg), global_step=global_step, writer=writer)

        if config.platform == "kaggle":
            model_filename = config.get_weight_file_path_kaggle("/kaggle/working/papers_implement/LLM/Transformer", f'{epoch:02d}')
        else:
            model_filename = config.get_weight_file_path(f'{epoch:02d}')

        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'global_step': global_step
        # }, model_filename)
    
    print(translate(model, 'Jane Eyre'))


def get_args():
    parser = argparse.ArgumentParser(description='Train Transformer')
    parser.add_argument('--n_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Number of epochs')
    parser.add_argument('--train_size', type=int, default=100, help='Training set size')
    parser.add_argument('--preload', type=str, default=None, help='Model weight')
    parser.add_argument('--platform', type=str, default=None, help='Platform used to train')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    config = Config()
    config.preload = args.preload
    config.platform = args.platform
    config.n_epochs = args.n_epochs
    config.train_size = args.train_size
    config.lr = args.lr
    assert config.d_k * config.n_head == config.d_model, f'd_k * n_head must equal to d_model'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_model(config, device)
    