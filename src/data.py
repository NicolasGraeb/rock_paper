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
    """Czyta gotowy podział z data_dir/{train,val,test}/<klasa>/ (tworzony przez load_dataset.py).

    Zwraca (loaders, class_names), gdzie loaders = {"train": ..., "val": ..., "test": ...}.
    """
    data_dir = Path(data_dir)
    missing = [s for s in SPLITS if not (data_dir / s).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Brak katalogów {missing} w {data_dir}, najpierw uruchom load_dataset.py"
        )

    train_tf = build_transforms(image_size=image_size, train=True, mean=mean, std=std)
    eval_tf = build_transforms(image_size=image_size, train=False, mean=mean, std=std)
    splits = {
        s: datasets.ImageFolder(root=str(data_dir / s), transform=train_tf if s == "train" else eval_tf)
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
            persistent_workers=num_workers > 0,  # na Windows tworzenie workerów co epokę jest kosztowne
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
    exclude: set[str] | None = None,
):
    """Loader do ewaluacji dowolnego splitu (test, external_dev, external_test, ...) z preprocessingiem checkpointu.

    Indeksy etykiet mapowane po NAZWIE klasy na indeksy checkpointu, bo zbiór może mieć podzbiór klas
    (np. zewnętrzny bez scissors), ale nie może mieć klas, których model nie zna.
    `exclude`: ścieżki względem data_dir (np. "external_dev/scissors/x.jpg") pomijane w ewaluacji.
    """
    root = Path(data_dir) / split
    if not root.is_dir():
        raise FileNotFoundError(f"Brak katalogu {root}")
    tf = build_transforms(preprocessing["image_size"], train=False, mean=preprocessing["mean"], std=preprocessing["std"])
    ds = datasets.ImageFolder(root=str(root), transform=tf)
    unknown = set(ds.classes) - set(class_names)
    if unknown:
        raise ValueError(f"Klasy {sorted(unknown)} w {root} nie występują w modelu ({class_names})")
    remap = {ds.class_to_idx[c]: class_names.index(c) for c in ds.classes}
    ds.samples = [(path, remap[t]) for path, t in ds.samples
                  if Path(path).relative_to(Path(data_dir)).as_posix() not in (exclude or set())]
    ds.targets = [t for _, t in ds.samples]
    ds.classes, ds.class_to_idx = list(class_names), {c: i for i, c in enumerate(class_names)}
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=torch.cuda.is_available())
    return loader
