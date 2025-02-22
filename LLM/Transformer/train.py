import os

import argparse
import torch

def argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_pkl", default=None)

    argparser.add_argument("--train_path", default=None)
    argparser.add_argument("--valid_path", default=None)

    argparser.add_argument("--epoch", type=str, default=10)
    argparser.add_argument("--batch_size", type=str, default=2048)
    argparser.add_argument("--n_warmup_steps", type=str, default=4000)

    argparser.add_argument("--d_model", type=str, default=512)
    argparser.add_argument("--d_inner_hid", type=str, default=2048)
    argparser.add_argument("--d_k", type=str, default=64)
    argparser.add_argument("--d_v", type=str, default=64)
    
    argparser.add_argument("--n_head", type=str, default=8)
    argparser.add_argument("--n_layers", type=str, default=6)

    argparser.add_argument("--dropout", type=float, default=0.1)

    argparser.add_argument("--output_dir", type=str, default=None)
    argparser.add_argument("--save_mode", type=str, choices=['all', 'best'], default="best")

    return argparser.parse_args()

if __name__ == "__main__":
    args = argparser()

    if not args.output_dir:
        print("No experiment result will be saved")
        raise

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load dataset
    if all((args.train_path, args.valid_path)):
        train_data, valid_data = prepare_dataloaders_from_bpe_files(args, device)
    elif args.data_pkl:
        train_data, valid_data = prepare_dataloaders_from_pkl(args, device)
    else:
        raise

    print(args)