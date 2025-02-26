import torch
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore

from utils.dataset import causual_mask

def greedy_decode(model, src, src_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    SOS_idx = tokenizer_tgt.token_to_id('[SOS]')
    EOS_idx = tokenizer_tgt.token_to_id('[EOS]')

    enc_output = model.encode(src, src_mask) # Precompute and reuse
    dec_input = torch.empty(1, 1).fill_(SOS_idx).type_as(src).to(device)
    while True:
        if dec_input.size(1) == max_len:
            break
        
        dec_mask = causual_mask(dec_input.size(1)).type_as(src_mask).to(device)

        out = model.decode(dec_input, enc_output, src_mask, dec_mask)

        # Get next token with max probabilities
        prob = model.project(out[:, -1])
        _, nxt_word = torch.max(prob, dim=1)

        dec_input = torch.cat([dec_input, torch.empty(1, 1).type_as(src).fill_(nxt_word.item()).to(device)], dim=1)

        if nxt_word == EOS_idx:
            break
        
    return dec_input.squeeze(0)

def run_validation(model, val_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    cnt = 0

    src_texts = []
    expected = []
    predicted = []

    console_width = 80
    with torch.no_grad():
        for batch in val_ds:
            cnt += 1

            # Input
            enc_input = batch['encoder_input'].to(device)
            assert enc_input.size(0) == 1, "Batch size must be 1"
            enc_mask = batch['encoder_mask'].to(device)
            model_out = greedy_decode(model, enc_input, enc_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            
            # Predict
            src_text = batch['src_text'][0]
            tgt_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            src_texts.append(src_text)
            expected.append(tgt_text)
            predicted.append(model_out_text)

            # Print to console
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{src_text}")
            print_msg(f"{f'TARGET: ':>12}{tgt_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if cnt == num_examples:
                break

    # Write to tensorboard
    if writer:
        # Character error rate
        metric = CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()
        
        # Word error rate
        metric = WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()
        
        # BLEU score
        metric = BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()