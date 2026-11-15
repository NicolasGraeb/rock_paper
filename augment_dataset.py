"""Tworzy kopię zbioru z augmentacją offline: train = oryginały + N zaugmentowanych kopii, val i test bez zmian."""
import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")

PRESETS = {
    "small": {
        "horizontal_flip": "p=0.5 (lewa/prawa dłoń)",
        "affine": "obrót ±15°, przesunięcie do 5%, skala 0.9-1.1",
        "color_jitter": "jasność 0.3, kontrast 0.3, nasycenie 0.3, odcień 0.05",
        "gaussian_blur": "p=0.2, kernel 5, sigma 0.1-1.5",
        "grayscale": "p=0.1",
    },
    "rot": {
        "rotation": "dowolny kąt -180° do 180°, rogi wypełnione odbiciem obrazu",
        "horizontal_flip": "p=0.5",
        "vertical_flip": "p=0.5",
        "affine": "przesunięcie do 5%, skala 0.9-1.1",
        "color_jitter": "jasność 0.3, kontrast 0.3, nasycenie 0.3, odcień 0.05",
        "gaussian_blur": "p=0.2, kernel 5, sigma 0.1-1.5",
        "grayscale": "p=0.1",
    },
}


class ReflectRotate:
    #obrót o losowy kąt bez czarnych rogów: pad odbiciem, obrót, przycięcie do oryginalnego rozmiaru
    def __init__(self, degrees: float):
        self.degrees = degrees

    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        pad = int(np.ceil(max(w, h) * 0.21)) + 2
        angle = float(torch.empty(1).uniform_(-self.degrees, self.degrees))
        img = F.pad(image, [pad], padding_mode="reflect")
        img = F.rotate(img, angle, interpolation=v2.InterpolationMode.BILINEAR)
        return F.center_crop(img, [h, w])


def build_augmentation(preset: str = "small"):
    color = [
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.2),
        v2.RandomGrayscale(p=0.1),
    ]
    if preset == "rot":
        return v2.Compose([
            ReflectRotate(180),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            *color,
        ])
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        *color,
    ])


def augment_one(job):
    src, dsts, seed, preset = job
    aug = build_augmentation(preset)
    image = Image.open(src).convert("RGB")
    for i, dst in enumerate(dsts):
        torch.manual_seed(seed + i)
        aug(image).save(dst, quality=95)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default=str(ROOT / "dataset" / "hagrid_user_split"))
    p.add_argument("--name", default="hagrid_user_split_aug")
    p.add_argument("--copies", type=int, default=2, help="zaugmentowanych kopii na każdy obraz train")
    p.add_argument("--preset", choices=list(PRESETS), default="small",
                   help="small: obrót ±15°, rot: dowolny obrót i odbicia w obu osiach")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.src)
    out_dir = src.parent / args.name
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_dir} już istnieje, użyj --overwrite")
        shutil.rmtree(out_dir)

    for s in SPLITS:
        shutil.copytree(src / s, out_dir / s)
    print(f"skopiowano {src} -> {out_dir}")

    split = pd.read_csv(src / "split.csv")
    split["augmented_from"] = None
    train = split[split["split"] == "train"].reset_index(drop=True)
    jobs, new_rows = [], []
    for idx, row in train.iterrows():
        rel = Path(row["path"])
        dsts = []
        for k in range(1, args.copies + 1):
            new_rel = rel.with_name(f"{rel.stem}_aug{k}.jpeg")
            dsts.append(out_dir / new_rel)
            new_rows.append({**row.to_dict(), "path": new_rel.as_posix(), "augmented_from": row["path"]})
        jobs.append((src / rel, dsts, args.seed * 1_000_003 + idx * args.copies, args.preset))

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        list(tqdm(ex.map(augment_one, jobs, chunksize=64), total=len(jobs), desc="augmentacja train"))

    split = pd.concat([split, pd.DataFrame(new_rows)], ignore_index=True)
    split.to_csv(out_dir / "split.csv", index=False)

    info = json.loads((src / "dataset_info.json").read_text(encoding="utf-8"))
    counts = split.groupby(["split", "label"]).size().unstack(fill_value=0)
    info.update({
        "name": out_dir.name, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parent": src.name, "transform": f"offline_aug_x{args.copies}" if args.preset == "small" else f"offline_aug_{args.preset}_x{args.copies}",
        "augmentation": PRESETS[args.preset], "augmentation_preset": args.preset, "augmentation_copies": args.copies, "augmentation_seed": args.seed,
        "n_images": len(split),
        "counts": {s: {c: int(v) for c, v in counts.loc[s].items()} for s in SPLITS},
    })
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(counts)
    print(f"zapisano: {out_dir}")


if __name__ == "__main__":
    main()
