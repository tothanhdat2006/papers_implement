from pathlib import Path
from configs.config import Config
from transformer.transformer import build_transformer 
from tokenizers import Tokenizer
from datasets import load_dataset
from utils.dataset import BilingualDataset
import torch
import sys

def translate(model, sentence: str):
    # Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    config = Config()
    tokenizer_src = Tokenizer.from_file(str(Path(config.tokenizer_file.format(config.lang_src))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config.tokenizer_file.format(config.lang_tgt))))

    # Load the pretrained weights
    if config.preload:
        if config.platform == "kaggle":
            model_filename = config.get_weight_file_path_kaggle("/kaggle/working/papers_implement/LLM/Transformer", config.preload)
        else:
            model_filename = config.get_weight_file_path(config.preload)
            
        if torch.cuda.is_available():
            state = torch.load(model_filename, weights_only=True)
        else:
            state = torch.load(model_filename, map_location=torch.device("cpu"), weights_only=True)

        model.load_state_dict(state['model_state_dict'])

    # if the sentence is a number use it as an index to the test set
    label = ""
    if type(sentence) == int or sentence.isdigit():
        id = int(sentence)
        ds = load_dataset(f"{config.datasource}", f"{config.lang_src}-{config.lang_tgt}", split='all')
        ds = BilingualDataset(ds, tokenizer_src, tokenizer_tgt, config.lang_src, config.lang_tgt, config.seq_len)
        sentence = ds[id]['src_text']
        label = ds[id]["tgt_text"]
    seq_len = config.seq_len

    # translate the sentence
    model.eval()
    with torch.no_grad():
        # Precompute the encoder output and reuse it for every generation step
        source = tokenizer_src.encode(sentence)
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
        ], dim=0).to(device)
        source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
        encoder_output = model.encode(source, source_mask)

        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)

        # Print the source sentence and target start prompt
        if label != "": print(f"{f'ID: ':>12}{id}") 
        print(f"{f'SOURCE: ':>12}{sentence}")
        if label != "": print(f"{f'TARGET: ':>12}{label}") 
        print(f"{f'PREDICTED: ':>12}", end='')

        # Generate the translation word by word
        while decoder_input.size(1) < seq_len:
            # build mask for target and calculate output
            decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # project next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

            # print the translated word
            print(f"{tokenizer_tgt.decode([next_word.item()])}", end=' ')

            # break if we predict the end of sentence token
            if next_word == tokenizer_tgt.token_to_id('[EOS]'):
                break

    # convert ids to tokens
    return tokenizer_tgt.decode(decoder_input[0].tolist())