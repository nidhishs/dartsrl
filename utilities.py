import torch
import glob
import os
import shutil
import logging
import sys
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import CIFAR10
from torchvision import transforms
from stable_baselines3.common.callbacks import BaseCallback
from model.genotypes import Genotype


def setup_logger(log_dir, name):
    """
    Setup logger.

    Args:
        log_dir: (str) log directory
        name: (str) name of the logger
    """
    log_format = "%(asctime)s [ %(levelname)8s ] %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt="%Y-%m-%dT%T%Z",
    )
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


def precision(output, target, topk=(1,)):
    """
    Computes precision@k for given batch.
    Percentage of samples for which ground truth is in top_k predicted classes.

    Args:
        output: (Tensor) model output tensor
        target: (Tensor) target tensor
        topk: (tuple) top-k values
    Returns:
        (list) precision@k values
    """
    max_k = max(topk)
    batch_size = target.size(0)

    _, top_k_predicted_classes = output.topk(max_k, dim=-1, largest=True, sorted=True)
    top_k_predicted_classes_t = top_k_predicted_classes.t()
    correct = top_k_predicted_classes_t == target.view(1, -1).expand_as(
        top_k_predicted_classes_t
    )

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdims=True)
        res.append((correct_k / batch_size))

    return res


def log_arguments(args, log_fn=print):
    """
    Log arguments.

    @param args: (argparse.Namespace) arguments
    @param log_fn: (function) logging function
    """
    separator = "+" + "-" * 27 + "+" + "-" * 47 + "+"

    log_fn(separator)
    log_fn(f'| {"Argument":^25} | {"Value":^45} |')
    log_fn(separator)
    for arg in vars(args):
        log_fn(f"| {str(arg).upper():<25} | {str(getattr(args, arg)):>45} |")
    log_fn(separator)


def get_transforms(dataset, train=True, cutout_length=None):
    """
    Get train and validation transforms.

    Args:
        dataset: (str) dataset name
        train: (bool) whether to use train or validation transform
        cutout_length: (int) length of the cutout mask
    Returns:
        (torchvision.transforms.Compose) train or validation transform
    """
    if dataset != "cifar10":
        raise NotImplementedError(f"{dataset} dataset is not implemented.")
    
    mean = [0.49139968, 0.48215827, 0.44653124]
    std = [0.24703233, 0.24348505, 0.26158768]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    if cutout_length is not None:
        train_transform.transforms.append(Cutout(cutout_length))
    
    return train_transform if train else valid_transform


def get_dataset(dataset, path, train=True, cutout_length=None):
    """
    Get dataset.

    Args:
        dataset: (str) dataset name
        path: (str) path to save dataset
    Returns:
        (torchvision.datasets) dataset
        (int) number of classes
        (int) number of input channels
    """
    if dataset != "cifar10":
        raise NotImplementedError(f"{dataset} dataset is not implemented.")

    num_classes = 10
    input_channels = 3

    if train:
        transform = get_transforms(dataset, train=True, cutout_length=cutout_length)
        train_data = CIFAR10(root=path, train=True, download=True, transform=transform)
        return train_data, num_classes, input_channels
    else:
        train_transform = get_transforms(dataset, train=True, cutout_length=cutout_length)
        valid_transform = get_transforms(dataset, train=False)
        train_data = CIFAR10(root=path, train=True, download=True, transform=train_transform)
        valid_data = CIFAR10(root=path, train=False, download=True, transform=valid_transform)
        return train_data, valid_data, num_classes, input_channels


def get_dataloader(train_data, valid_data, batch_size, num_workers=4):
    """
    Get train and validation dataloaders.

    Args:
        train_data: (torch.utils.data.Dataset) train dataset
        valid_data: (torch.utils.data.Dataset) validation dataset
        batch_size: (int) batch size
        num_workers: (int) number of workers
    Returns:
        (torch.utils.data.DataLoader) train dataloader
        (torch.utils.data.DataLoader) validation dataloader
    """
    if valid_data is None:
        train_data, valid_data = torch.utils.data.random_split(train_data, [0.5, 0.5])

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, valid_loader


def stash_files(log_dir):
    """
    Stash all python experiment files to the log_dir.

    Args:
        log_dir: (str) log directory
    """
    py_files = glob.glob("**/*.py", recursive=True)
    stash_dir = os.path.join(log_dir, "stash")
    os.makedirs(stash_dir, exist_ok=True)
    for py_file in py_files:
        base_dir = os.path.basename(os.path.dirname(py_file))
        dest_dir = os.path.join(stash_dir, base_dir)
        os.makedirs(os.path.join(stash_dir, base_dir), exist_ok=True)
        shutil.copy(py_file, dest_dir)


def measure(log_fn=print):
    """
    Decorator for measuring stats.

    Args:
        log_fn: (function) logging function
    Returns:
        (function) wrapped function
    """
    def decorator(func):
        def get_gpu_usage(pid):
            gpu = os.popen(f'nvidia-smi --query-compute-apps=pid,used_memory --format=csv | grep {pid}').read().strip()
            return gpu.split(', ')[-1] if gpu else '0 MiB'
        
        def get_time_elapsed(start_time):
            elapsed = time.time() - start_time
            hours, rem = divmod(elapsed, 3600)
            minutes, seconds = divmod(rem, 60)
            return f"{int(hours):2d}h {int(minutes):02d}m {int(seconds):02d}s"

        def wrapper(*args, **kwargs):
            tic = time.time()
            result = func(*args, **kwargs)
            pid = os.getpid()

            if torch.cuda.is_available():
                log_fn(f"GPU usage: {get_gpu_usage(pid)}")
            log_fn(f"Time elapsed: {get_time_elapsed(tic)}")
            return result
        
        return wrapper
    return decorator


def get_gpu_usage(pid):
    """
    Get GPU usage for given process id.

    Args:
        pid: (int) process id
    Returns:
        (str) GPU usage
    """
    gpu = os.popen(f'nvidia-smi --query-compute-apps=pid,used_memory --format=csv | grep {pid}').read().strip()
    return gpu.split(', ')[-1] if gpu else '0 MiB'


class AverageMeter:
    def __init__(self):
        """
        Computes and stores the average and current value.
        """
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def reset(self):
        """
        Reset all values.
        """
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, value, n=1):
        """
        Update average and current value.

        Args:
            value: (float) current value
            n: (int) number of samples
        """
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


class Cutout:
    def __init__(self, length):
        """
        Cutout augmentation.

        Args:
            length: (int) length of the cutout mask
        """
        self.length = length

    def __call__(self, img):
        """
        Args:
            img: (torch.Tensor) image tensor
        Returns:
            (torch.Tensor) image tensor with cutout mask
        """
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img
    

class LogGenotypeCallback(BaseCallback):
    def _on_step(self) -> bool:
        # if self.training_env.env_method("get_genotype")[0] is None:
        #     return True
        
        # genotype = self.training_env.env_method("get_genotype")[0]
        # self.logger.log("genotype: ", genotype)
        # return True

        print(self.num_timesteps, self.training_env.env_method("is_done"))
