from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.transforms import IMAGENET_MEAN, IMAGENET_STD, build_transforms

SPLITS = ("train", "val", "test")


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    seed: int = 42,
    num_workers: int = 0,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
):
    data_dir = Path(data_dir)
    missing = [s for s in SPLITS if not (data_dir / s).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Brak katalogów {missing} w {data_dir}, najpierw uruchom load_dataset.py"
        )

    tf = build_transforms(image_size=image_size, mean=mean, std=std)
    splits = {
        s: datasets.ImageFolder(root=str(data_dir / s), transform=tf)
        for s in SPLITS
    }
    class_names = list(splits["train"].classes)
    for s, ds in splits.items():
        if ds.classes != class_names:
            raise ValueError(f"Klasy w {s} ({ds.classes}) różnią się od train ({class_names})")

    loaders = {
        s: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=s == "train",
            generator=torch.Generator().manual_seed(seed) if s == "train" else None,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )
        for s, ds in splits.items()
    }
    return loaders, class_names


def create_eval_loader(
    data_dir: str | Path,
    split: str,
    class_names: list[str],
    preprocessing: dict,
    batch_size: int = 32,
    num_workers: int = 0,
):
    root = Path(data_dir) / split
    if not root.is_dir():
        raise FileNotFoundError(f"Brak katalogu {root}")
    tf = build_transforms(preprocessing["image_size"], mean=preprocessing["mean"], std=preprocessing["std"])
    ds = datasets.ImageFolder(root=str(root), transform=tf)
    if ds.classes != list(class_names):
        raise ValueError(f"Klasy w {root} ({ds.classes}) różnią się od klas modelu ({class_names})")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=torch.cuda.is_available())
    return loader
