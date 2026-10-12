from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from src.transforms import build_transforms


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 128,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
):
    data_dir = Path(data_dir)
    train_tf = build_transforms(image_size=image_size, train=True)
    eval_tf = build_transforms(image_size=image_size, train=False)

    probe = datasets.ImageFolder(root=str(data_dir), transform=eval_tf)
    class_names = list(probe.classes)
    n = len(probe)
    n_val = max(1, int(round(n * val_ratio)))
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_ds = datasets.ImageFolder(root=str(data_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(root=str(data_dir), transform=eval_tf)
    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, class_names
