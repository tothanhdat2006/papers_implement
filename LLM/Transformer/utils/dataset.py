import torch
from torch.utils.data import Dataset

def causual_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.SOS_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.EOS_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.PAD_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Text -> token -> IDs
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Fill <PAD> token to fit length (seq_len)
        enc_num_paddding_tokens = self.seq_len - len(enc_input_tokens) - 2 # SOS and EOS
        dec_num_paddding_tokens = self.seq_len - len(dec_input_tokens) - 1 # Only one

        assert enc_num_paddding_tokens >= 0 and dec_num_paddding_tokens >= 0, f"Sentecne is too long"

        enc_input = torch.cat(
            [
                self.SOS_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.EOS_token,
                torch.tensor([self.PAD_token] * enc_num_paddding_tokens, dtype=torch.int64)
            ]
        )

        dec_input = torch.cat(
            [
                self.SOS_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.PAD_token] * dec_num_paddding_tokens, dtype=torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.EOS_token,
                torch.tensor([self.PAD_token] * dec_num_paddding_tokens, dtype=torch.int64)
            ]
        )

        assert enc_input.size(0) == self.seq_len
        assert dec_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'encoder_input': enc_input,
            'decoder_input': dec_input,
            'encoder_mask': (enc_input != self.PAD_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            'decoder_mask': (dec_input != self.PAD_token).unsqueeze(0).int() & causual_mask(dec_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text,
        }
        


