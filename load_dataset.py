"""Pobiera HaGRID i buduje dataset/hagrid_user_split (podział po user_id, ~70/15/15).

    uv run python load_dataset.py
"""
import argparse
import io
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
THUMB_WH = (48, 32)

HAGRID_REPO = "cj-mills/hagrid-classification-512p-no-gesture-150k"
HAGRID_TREE = f"https://huggingface.co/api/datasets/{HAGRID_REPO}/tree/main/data"
HAGRID_RESOLVE = f"https://huggingface.co/datasets/{HAGRID_REPO}/resolve/main/"
#etykieta HaGRID -> (gest, klasa)
HAGRID_LABELS = {2: ("fist", "rock"), 9: ("palm", "paper"), 10: ("peace", "scissors")}
HAGRID_ANNOTATIONS = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", default=str(ROOT / "data_cache" / "hagrid"))
    p.add_argument("--annotations-dir", default=str(ROOT / "data_cache" / "hagrid_annotations"))
    p.add_argument("--name", default="hagrid_user_split")
    p.add_argument("--out-root", default=str(ROOT / "dataset"))
    p.add_argument("--per-class", type=int, default=None, help="obrazów na klasę (domyślnie liczność najmniejszej)")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--group-threshold", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", help="zbuduj dataset/<name> od nowa")
    return p.parse_args()


def thumbnail(image: Image.Image) -> np.ndarray:
    return cv2.resize(np.array(image.convert("L")), THUMB_WH, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()


def normalize(t: np.ndarray) -> np.ndarray:
    t = t - t.mean(1, keepdims=True)
    return t / np.linalg.norm(t, axis=1, keepdims=True).clip(1e-8)


def groups_from_thumbnails(thumbs: np.ndarray, threshold: float, chunk: int = 2048) -> np.ndarray:
    thumbs = normalize(thumbs)
    rows, cols = [], []
    for start in range(0, len(thumbs), chunk):
        r, c = np.nonzero(thumbs[start:start + chunk] @ thumbs.T >= threshold)
        keep = r + start != c
        rows.append(r[keep] + start)
        cols.append(c[keep])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    adj = csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(len(thumbs), len(thumbs)))
    return connected_components(adj, directed=False)[1]


def merge_groups(*groupings) -> np.ndarray:
    n, rows, cols, offset = len(groupings[0]), [], [], len(groupings[0])
    for g in groupings:
        codes = pd.factorize(pd.Series(g))[0]
        rows.append(np.arange(n))
        cols.append(codes + offset)
        offset += codes.max() + 1
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    adj = csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(offset, offset))
    return connected_components(adj, directed=False)[1][:n]


def take_group_fold(labels, groups, ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    sgkf = StratifiedGroupKFold(n_splits=max(2, round(1 / ratio)), shuffle=True, random_state=seed)
    return next(sgkf.split(np.zeros(len(labels)), labels, groups))


def assign_train_val_test(df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int) -> pd.Series:
    labels, groups = df["label"].to_numpy(str), df["group"].to_numpy()
    split = np.empty(len(df), dtype=object)
    rest, test = take_group_fold(labels, groups, test_ratio, seed)
    split[test] = "test"
    train_rel, val_rel = take_group_fold(labels[rest], groups[rest], val_ratio / (1 - test_ratio), seed)
    split[rest[val_rel]] = "val"
    split[rest[train_rel]] = "train"
    return pd.Series(split, index=df.index)


def download(url: str, dst: Path, size: int | None = None):
    if dst.exists() and (size is None or dst.stat().st_size == size):
        return
    print(f"pobieram {dst.name}" + (f" ({size / 1e6:.0f} MB)" if size else ""), flush=True)
    tmp = dst.with_suffix(".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as out:
        shutil.copyfileobj(r, out, length=1 << 20)
    tmp.replace(dst)


def hagrid_user_ids(ann_dir: Path) -> dict[str, str]:
    gestures = [g for g, _ in HAGRID_LABELS.values()]
    jsons = [ann_dir / "annotations" / s / f"{g}.json" for s in SPLITS for g in gestures]
    if not all(p.exists() for p in jsons):
        archive = ann_dir / "annotations.zip"
        ann_dir.mkdir(parents=True, exist_ok=True)
        download(HAGRID_ANNOTATIONS, archive)
        with zipfile.ZipFile(archive) as z:
            for p in jsons:
                z.extract(p.relative_to(ann_dir).as_posix(), ann_dir)
    users = {}
    for p in jsons:
        users.update({uuid: a["user_id"] for uuid, a in json.loads(p.read_text(encoding="utf-8")).items()})
    return users


def read_hagrid(cache: Path) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(HAGRID_TREE) as r:
        files = sorted((f for f in json.load(r) if f["path"].endswith(".parquet")), key=lambda f: f["path"])
    rows = []
    for f in files:
        shard = cache / Path(f["path"]).name
        download(HAGRID_RESOLVE + f["path"], shard, f["size"])
        table = pq.read_table(shard)
        labels = table.column("label").to_numpy()
        idx = np.nonzero(np.isin(labels, list(HAGRID_LABELS)))[0]
        for i, img in zip(idx, table.column("image").take(idx).to_pylist()):
            gesture, label = HAGRID_LABELS[int(labels[i])]
            rows.append({"bytes": img["bytes"], "label": label, "gesture": gesture,
                         "file": f"{shard.stem.split('-')[1]}_{i:06d}{Path(img.get('path') or '.jpg').suffix or '.jpg'}",
                         "source": f"{shard.name}#{i}", "uuid": Path(img.get("path") or "").stem})
        print(f"{shard.name}: {len(idx)} obrazów z wybranych gestów", flush=True)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    out_dir = Path(args.out_root) / args.name
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_dir} już istnieje, użyj --overwrite, żeby zbudować go od nowa")
        shutil.rmtree(out_dir)

    df = read_hagrid(Path(args.cache_dir))
    df["user_id"] = df["uuid"].map(hagrid_user_ids(Path(args.annotations_dir)))
    print(f"bez user_id w adnotacjach: {df['user_id'].isna().sum()} obrazów, pominięte")
    df = df.dropna(subset=["user_id"]).reset_index(drop=True)
    print("przed balansowaniem:", df["label"].value_counts().sort_index().to_dict())
    n = min(args.per_class or np.inf, int(df["label"].value_counts().min()))
    df = df.groupby("label").sample(n=n, random_state=args.seed).reset_index(drop=True)
    print(f"po balansowaniu: {n} na klasę, razem {len(df)}")

    thumbs = np.stack([thumbnail(Image.open(io.BytesIO(b))) for b in df["bytes"]])
    df["group"] = merge_groups(groups_from_thumbnails(thumbs, args.group_threshold), df["user_id"])
    df["group_size"] = df["group"].map(df["group"].value_counts())
    print(f"{df['user_id'].nunique()} osób, {df['group'].nunique()} grup")
    df["split"] = assign_train_val_test(df, args.val_ratio, args.test_ratio, args.seed)
    assert df.groupby("group")["split"].nunique().max() == 1, "grupa rozdzielona między zbiory"
    assert df.groupby("user_id")["split"].nunique().max() == 1, "osoba w więcej niż jednym splicie"
    df["path"] = [f"{s}/{label}/{f}" for s, label, f in zip(df["split"], df["label"], df["file"])]

    for data, rel in zip(df["bytes"], df["path"]):
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (out_dir / rel).write_bytes(data)
    df[["path", "split", "label", "gesture", "group", "group_size", "user_id", "source"]].to_csv(
        out_dir / "split.csv", index=False)
    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(SPLITS, fill_value=0)
    info = {
        "name": out_dir.name, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": f"hf:{HAGRID_REPO}", "label_map": {label: g for g, label in HAGRID_LABELS.values()},
        "parent": None, "transform": None, "seed": args.seed, "per_class": n, "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio, "group_threshold": args.group_threshold, "group_by": "user",
        "n_groups": int(df["group"].nunique()), "n_users": int(df["user_id"].nunique()), "n_images": len(df),
        "counts": {s: {c: int(v) for c, v in counts.loc[s].items()} for s in SPLITS},
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    print(counts.assign(razem=counts.sum(axis=1), udział=(counts.sum(axis=1) / len(df)).round(3)))
    print(f"zapisano: {out_dir}")


if __name__ == "__main__":
    main()
