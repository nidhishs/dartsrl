import argparse
import logging
import os
import time
import sys

import numpy as np
import torch
import torch.nn as nn
import utilities as u

from torch.utils.tensorboard import SummaryWriter
from model import SearchNetwork, SEARCH_SPACES


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="dartsrl-wgt", help="Name of the experiment.")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to use for training.")
    parser.add_argument("--data-path", type=str, default="./data", help="Path to dataset.")
    parser.add_argument("--space", type=str, default="darts", help="darts/s2/s3/s4/nas-bench-201")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs for the super network.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for train and valid dataloaders.")
    parser.add_argument("--layers", type=int, default=8, help="Number of layers in the super network.")
    parser.add_argument("--initial-channels", type=int, default=16, help="Initial number of channels after input channels in super network.")
    parser.add_argument("--lr", type=float, default=0.025, help="Learning rate for super network SGD optimizer.")
    parser.add_argument("--min-lr", type=float, default=0.001, help="Minimum learning rate for super network cosine-annealing scheduler.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for super network SGD optimizer.")
    parser.add_argument("--weight-decay", type=float, default=3e-4, help="Weight decay for super network SGD optimizer.")
    parser.add_argument("--grad-clip", type=float, default=5, help="Gradient clipping value.")
    parser.add_argument("--cutout", action="store_true", default=False, help="Use cutout augmentation.")
    parser.add_argument("--cutout-length", type=int, default=16, help="Cutout length.")
    parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU backend.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Path to save logs.")
    parser.add_argument("--log-interval", type=int, default=50, help="Log frequency.")
    args = parser.parse_args()
    args.job_id = f"{args.name}_{int(time.time())}"
    args.log_dir = os.path.join(args.log_dir, args.job_id)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.cutout_length = args.cutout_length if args.cutout else None
    os.makedirs(args.log_dir, exist_ok=True)
    del args.cpu, args.cutout

    return args


@u.measure(log_fn=logging.info)
def main(args):
    writer = SummaryWriter(args.log_dir)

    # ====== Dataset ====== #
    dataset, num_classes, input_channels = u.get_dataset(
        args.dataset, args.data_path, train=True, cutout_length=args.cutout_length
    )
    train_loader, valid_loader = u.get_dataloader(dataset, None, args.batch_size)
    # ===================== #

    # ====== Model ====== #
    model = SearchNetwork(
        c_in=input_channels, c=args.initial_channels, num_classes=num_classes,
        num_layers=args.layers, primitives=SEARCH_SPACES[args.space]
    ).to(args.device)
    optimizer = torch.optim.SGD(
        model.weights, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.min_lr)
    criterion = nn.CrossEntropyLoss().to(args.device)
    # =================== #

    # ====== Training ====== #
    best_top1 = 0
    for epoch in range(args.epochs):
        t_top1, t_top5, t_loss = train(model, train_loader, optimizer, criterion, args.grad_clip, args.device, args.log_interval)
        v_top1, v_top5, v_loss = valid(model, valid_loader, criterion, args.device, args.log_interval)
        scheduler.step()

        logging.info(
            f"train-epoch : " + f"[ {f'{epoch+1:2d}/{args.epochs}':^5} ] " +
            f'Final Train Loss = {t_loss:.3f}, Final Train Prec@(1, 5) = ({t_top1:.1%}, {t_top5:.1%})'
        )
        logging.info(
            f"valid-epoch : " + f"[ {f'{epoch+1:2d}/{args.epochs}':^5} ] " +
            f'Final Valid Loss = {v_loss:.3f}, Final Valid Prec@(1, 5) = ({v_top1:.1%}, {v_top5:.1%})'
        )

        if writer is not None:
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
            writer.add_scalar("train/loss", t_loss, epoch)
            writer.add_scalar("train/top1", t_top1, epoch)
            writer.add_scalar("train/top5", t_top5, epoch)
            writer.add_scalar("valid/loss", v_loss, epoch)
            writer.add_scalar("valid/top1", v_top1, epoch)
            writer.add_scalar("valid/top5", v_top5, epoch)
        
        if v_top1 > best_top1:
            best_top1 = v_top1
            torch.save(model.state_dict(), os.path.join(args.log_dir, "model.pt"))
    # ====================== #

    logging.info(f"Best top1 accuracy: {best_top1:.2%}")
    logging.info(f"Logs saved at {args.log_dir}")


def train(
        model, train_loader, optimizer, criterion, grad_clip, device="cuda", log_interval=50
    ):
    """
    Train the model for one epoch.

    Args:
        model: (nn.Module) model to train
        train_loader: (DataLoader) training data loader
        optimizer: (torch.optim.Optimizer) optimizer
        criterion: (nn.Module) loss function
        grad_clip: (float) gradient clipping value
        device: (str) torch device
        log_interval: (int) log frequency
    Returns:
        (float) top1 accuracy 
        (float) top5 accuracy
        (float) loss
    """
    model.train()
    top1, top5, losses = u.AverageMeter(), u.AverageMeter(), u.AverageMeter()

    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        minibatch_size = x.size(0)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.weights, grad_clip)
        optimizer.step()
        prec_1, prec_5 = u.precision(logits, y, topk=(1, 5))

        top1.update(prec_1.item(), minibatch_size)
        top5.update(prec_5.item(), minibatch_size)
        losses.update(loss.item(), minibatch_size)
        # ====== Logging ====== #
        if step % log_interval == 0 or step == len(train_loader) - 1:
            logging.info(
                f"train-step  : [ {f'{step:03d}/{len(train_loader) - 1:03d}':^10} ] " +
                f'Loss = {losses.avg:.3f}, Prec@(1, 5) = ({top1.avg:.2%}, {top5.avg:.2%})'
            )
        # ===================== #
    return top1.avg, top5.avg, losses.avg


@torch.no_grad()
def valid(
        model, valid_loader, criterion, device="cuda", log_interval=50
    ):
    """
    Validate the model.

    Args:
        model: (nn.Module) model to validate
        valid_loader: (DataLoader) validation data loader
        criterion: (nn.Module) loss function
        device: (str) torch device
        log_interval: (int) log frequency
    Returns:
        (float) top1 accuracy
        (float) top5 accuracy
        (float) loss
    """
    model.eval()
    top1, top5, losses = u.AverageMeter(), u.AverageMeter(), u.AverageMeter()

    for step, (x, y) in enumerate(valid_loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        minibatch_size = x.size(0)
        
        logits = model(x)
        loss = criterion(logits, y)
        prec_1, prec_5 = u.precision(logits, y, topk=(1, 5))

        top1.update(prec_1.item(), minibatch_size)
        top5.update(prec_5.item(), minibatch_size)
        losses.update(loss.item(), minibatch_size)
        # ====== Logging ====== #
        if step % log_interval == 0 or step == len(valid_loader) - 1:
            logging.info(
                f"valid-step  : [ {f'{step:03d}/{len(valid_loader) - 1:03d}':^10} ] " +
                f'Loss = {losses.avg:.3f}, Prec@(1, 5) = ({top1.avg:.2%}, {top5.avg:.2%})'
            )
        # ===================== #
    return top1.avg, top5.avg, losses.avg


if __name__ == "__main__":
    arguments = get_arguments()
    u.setup_logger(arguments.log_dir, arguments.name)
    u.stash_files(arguments.log_dir)
    logging.info(f"python3 {' '.join(sys.argv)}")
    u.log_arguments(arguments, logging.info)

    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)

    main(arguments)
