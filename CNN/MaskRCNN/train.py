import os
import re
import bisect
import glob
import sys
import time
import argparse

import torch

from model.mask_rcnn import make_maskrcnn
from datasets.coco_eval import CocoEvaluator,prepare_for_coco
from datasets.utils import datasets
from utils.gpu import *
from utils.utils import Meter, TextArea, save_ckpt


def train_one_epoch(model, optimizer, data_loader, device, epoch, args):
    for p in optimizer.param_groups:
        p["lr"] = args.lr_epoch

    iters = len(data_loader) if args.iters < 0 else args.iters

    t_m = Meter("total")
    m_m = Meter("model")
    b_m = Meter("backward")

    model.train()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()
        num_iters = epoch * len(data_loader) + i
        if num_iters <= args.warmup_iters:
            r = num_iters / args.warmup_iters
            for j, p in enumerate(optimizer.param_groups):
                p["lr"] = r * args.lr_epoch
                   
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}
        S = time.time()
        
        losses = model(image, target)
        total_loss = sum(losses.values())
        m_m.update(time.time() - S)
            
        S = time.time()
        total_loss.backward()
        b_m.update(time.time() - S)
        
        optimizer.step()
        optimizer.zero_grad()

        if num_iters % args.print_freq == 0:
            print("{}\t".format(num_iters), "\t".join("{:.3f}".format(l.item()) for l in losses.values()))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
           
    A = time.time() - A
    print("Time per iter: {:.1f}, Total: {:.1f}, Model: {:.1f}, Backward: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg,1000*b_m.avg))
    return A / iters
            

def evaluate(model, data_loader, device, args, generate=True):
    iter_eval = None
    if generate:
        iter_eval = generate_results(model, data_loader, device, args)

    dataset = data_loader #
    iou_types = ["bbox", "segm"]
    coco_evaluator = CocoEvaluator(dataset.coco, iou_types)

    results = torch.load(args.results, map_location="cpu")

    S = time.time()
    coco_evaluator.accumulate(results)
    print("accumulate: {:.1f}s".format(time.time() - S))

    # collect outputs of buildin function print
    temp = sys.stdout
    sys.stdout = TextArea()

    coco_evaluator.summarize()

    output = sys.stdout
    sys.stdout = temp
        
    return output, iter_eval
    
    
# generate results file   
@torch.no_grad()   
def generate_results(model, data_loader, device, args):
    iters = len(data_loader) if args.iters < 0 else args.iters
        
    t_m = Meter("total")
    m_m = Meter("model")
    coco_results = []
    model.eval()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()
        
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}

        S = time.time()
        #torch.cuda.synchronize()
        output = model(image)
        m_m.update(time.time() - S)
        
        prediction = {target["image_id"].item(): {k: v.cpu() for k, v in output.items()}}
        coco_results.extend(prepare_for_coco(prediction))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
     
    A = time.time() - A 
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg))
    torch.save(coco_results, args.results)
        
    return A / iters
    
def train_mode(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if device.type == "cuda": 
        get_gpu_prop(show=True)
    print(f"\nDevice: {device}")
        
    # ---------------------- Prepare data loader ------------------------------- #
    
    dataset_train = datasets(args.dataset, args.data_dir, "train", train=True)
    indices = torch.randperm(len(dataset_train)).tolist()
    d_train = torch.utils.data.Subset(dataset_train, indices)
    
    d_test = datasets(args.dataset, args.data_dir, "val", train=True) # set train=True for eval
        
    args.warmup_iters = max(1000, len(d_train))
    
    # -------------------------------------------------------------------------- #

    print(args)
    num_classes = max(d_train.dataset.classes) + 1 # including background class
    model = make_maskrcnn(num_classes, True).to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_lambda = lambda x: 0.1 ** bisect.bisect(args.lr_steps, x)
    
    start_epoch = 0
    
    # Find all checkpoints, and load the latest checkpoint
    prefix, ext = os.path.splitext(args.ckpt_path)
    ckpts = glob.glob(prefix + "-*" + ext)
    ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
    if ckpts:
        checkpoint = torch.load(ckpts[-1], map_location=device) # load last checkpoint
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epochs"]

        del checkpoint
        torch.cuda.empty_cache()

    since = time.time()
    print(f"\nLoaded checkpoint: already trained {start_epoch} epochs; to {args.epochs} epochs")
    
    # ------------------------------- Training ------------------------------------ #
        
    for epoch in range(start_epoch, args.epochs):
        print("\nEpoch: {}".format(epoch + 1))
            
        A = time.time()
        args.lr_epoch = lr_lambda(epoch) * args.lr
        print("Hyperparameter lr_epoch: {:.5f}, factor: {:.5f}".format(args.lr_epoch, lr_lambda(epoch)))
        iter_train = train_one_epoch(model, optimizer, d_train, device, epoch, args)
        A = time.time() - A
        
        B = time.time()
        eval_output, iter_eval = evaluate(model, d_test, device, args)
        B = time.time() - B

        trained_epoch = epoch + 1
        print("Training: {:.1f} s, Evaluation: {:.1f} s".format(A, B))
        collect_gpu_info("maskrcnn", [1 / iter_train, 1 / iter_eval])
        print(eval_output.get_AP())

        if epoch % 5 == 0:
            save_ckpt(model, optimizer, trained_epoch, args.ckpt_path, eval_info=str(eval_output))

        # it will create many checkpoint files during training, so delete some.
        # prefix, ext = os.path.splitext(args.ckpt_path)
        # ckpts = glob.glob(prefix + "-*" + ext)
        # ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
        # n = 2
        # if len(ckpts) > n:
        #     for i in range(len(ckpts) - n):
        #         os.system("rd {}".format(ckpts[i]))

    print("\nTotal time of this training: {:.1f} s".format(time.time() - since))
    if start_epoch < args.epochs:
        print("Already trained: {} epochs\n".format(trained_epoch))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cuda", action="store_true")
    
    parser.add_argument("--dataset", default="voc", help="coco or voc")
    parser.add_argument("--data-dir", default="D:/MachineLearning/Datasets/VOC2012")
    parser.add_argument("--ckpt-path")
    parser.add_argument("--results")
    
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument('--lr-steps', nargs="+", type=int, default=[6, 7])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10, help="Max iters per epoch, -1 denotes auto")
    parser.add_argument("--print-freq", type=int, default=100, help="Frequency of printing losses")
    return parser.parse_args()
    
if __name__ == "__main__":
    args = get_args()
    
    if args.lr is None:
        args.lr = 0.02 * 1 / 16 # lr should be 'batch_size / 16 * 0.02'

    if args.ckpt_path is None:
        args.ckpt_path = "./maskrcnn_{}.pth".format(args.dataset)
    if args.results is None:
        args.results = os.path.join(os.path.dirname(args.ckpt_path), "maskrcnn_results.pth")
    
    train_mode(args)
    
    