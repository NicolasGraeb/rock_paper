"""Pobiera / buduje zbiory danych projektu w dataset/<nazwa>/.

    uv run python load_dataset.py hagrid          # główny zbiór: HaGRID z HF (~3.8 GB) -> dataset/dataset_origin_new/{train,val,test}
    uv run python load_dataset.py hand-in-wild    # zewnętrzny test: Zenodo (~650 MB) -> dataset/external_hiw/{external_dev,external_test}
    uv run python load_dataset.py folder --source moje_zdjecia --name external_own   # własne zdjęcia: <klasa>/[<osoba>/]*.jpg
    uv run python load_dataset.py kaggle          # stary zbiór drgfreeman rozpakowany w archive/ -> dataset/dataset_origin

Każdy zbiór dostaje split.csv (który plik w którym splicie) i dataset_info.json (źródło, seed, liczności).

train/val/test (hagrid, kaggle): podział grupowy — serie prawie identycznych zdjęć (korelacja miniatur 48x32
>= --group-threshold) w całości w jednym splicie — i stratyfikowany, ~70/15/15.

external (hand-in-wild, folder): tylko do ewaluacji, nigdy do treningu. EXIF orientation zapisywana na stałe,
dokładne duplikaty usuwane, obrazy pokrywające się z --train-dir (MD5 / korelacja miniatur) odrzucane,
podział external_dev / external_test ~50/50 po osobach/sesjach (podkatalog albo sesja z EXIF).
preview_<klasa>.jpg służy do ręcznego sprawdzenia, czy gesty pasują do etykiet.
"""
import argparse
import hashlib
import io
import json
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import ExifTags, Image, ImageOps
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
CLASSES = ("paper", "rock", "scissors")
SPLITS = ("train", "val", "test")
EXTERNAL_SPLITS = ("external_dev", "external_test")
IMG_EXT = {".png", ".jpg", ".jpeg"}
THUMB_WH = (48, 32)

HAGRID_REPO = "cj-mills/hagrid-classification-512p-no-gesture-150k"
HAGRID_TREE = f"https://huggingface.co/api/datasets/{HAGRID_REPO}/tree/main/data"
HAGRID_RESOLVE = f"https://huggingface.co/datasets/{HAGRID_REPO}/resolve/main/"
# indeks etykiety w HaGRID -> (gest, klasa w projekcie); uwaga: HaGRID ma też własną klasę "rock" (indeks 12, „rogi”)
HAGRID_LABELS = {2: ("fist", "rock"), 9: ("palm", "paper"), 10: ("peace", "scissors")}
HIW_RECORD = "https://zenodo.org/api/records/5503489"
EXIF_DATETIME = 36867  # DateTimeOriginal (w Exif IFD)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="source", required=True)

    hagrid = sub.add_parser("hagrid", help="HaGRID z Hugging Face (fist/palm/peace)")
    hagrid.add_argument("--cache-dir", default=str(ROOT / "data_cache" / "hagrid"))
    hagrid.add_argument("--name", default="dataset_origin_new")
    hagrid.add_argument("--per-class", type=int, default=None, help="obrazów na klasę (domyślnie liczność najmniejszej)")

    kaggle = sub.add_parser("kaggle", help="drgfreeman/rockpaperscissors rozpakowany w archive/")
    kaggle.add_argument("--archive", default=str(ROOT / "archive"))
    kaggle.add_argument("--name", default="dataset_origin")

    hiw = sub.add_parser("hand-in-wild", help="Hand_in_Wild z Zenodo (zbiór zewnętrzny)")
    hiw.add_argument("--cache-dir", default=str(ROOT / "data_cache" / "hand_in_wild"))
    hiw.add_argument("--name", default="external_hiw")

    folder = sub.add_parser("folder", help="własne zdjęcia (zbiór zewnętrzny)")
    folder.add_argument("--dir", required=True, help="katalog z podkatalogami klas")
    folder.add_argument("--name", required=True)

    for sp in (hagrid, kaggle):
        sp.add_argument("--val-ratio", type=float, default=0.15)
        sp.add_argument("--test-ratio", type=float, default=0.15)
        sp.add_argument("--group-threshold", type=float, default=0.90)
    for sp in (hiw, folder):
        sp.add_argument("--classes", nargs="+", choices=CLASSES, default=list(CLASSES))
        sp.add_argument("--train-dir", default=str(ROOT / "dataset" / "dataset_origin_new"),
                        help="zbiór treningowy, z którym sprawdzane jest nakładanie")
        sp.add_argument("--overlap-threshold", type=float, default=0.95)
        sp.add_argument("--max-side", type=int, default=1024)
        sp.add_argument("--session-gap-min", type=float, default=5.0)
        sp.add_argument("--dev-ratio", type=float, default=0.5)
    for sp in (hagrid, kaggle, hiw, folder):
        sp.add_argument("--out-root", default=str(ROOT / "dataset"))
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--overwrite", action="store_true", help="zbuduj dataset/<name> od nowa")
    return p.parse_args()


# ---------- wspólne ----------

def thumbnail(image: Image.Image) -> np.ndarray:
    return cv2.resize(np.array(image.convert("L")), THUMB_WH, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()


def normalize(t: np.ndarray) -> np.ndarray:
    t = t - t.mean(1, keepdims=True)  # korelacja Pearsona = cosinus po odjęciu średniej
    return t / np.linalg.norm(t, axis=1, keepdims=True).clip(1e-8)


def groups_from_thumbnails(thumbs: np.ndarray, threshold: float, chunk: int = 2048) -> np.ndarray:
    """Spójne składowe grafu "korelacja miniatur >= threshold". Liczone w kawałkach (duże zbiory)."""
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


def take_group_fold(labels, groups, ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Odcina ~ratio próbek (całymi grupami, ze stratyfikacją). Zwraca (indeksy reszty, indeksy odcięte)."""
    sgkf = StratifiedGroupKFold(n_splits=max(2, round(1 / ratio)), shuffle=True, random_state=seed)
    return next(sgkf.split(np.zeros(len(labels)), labels, groups))


def assign_train_val_test(df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int) -> pd.Series:
    labels, groups = df["label"].to_numpy(str), df["group"].to_numpy()
    split = np.empty(len(df), dtype=object)
    rest, test = take_group_fold(labels, groups, test_ratio, seed)
    split[test] = "test"
    # val liczony jako udział w całym zbiorze, więc w pozostałej części jest odpowiednio większy
    train_rel, val_rel = take_group_fold(labels[rest], groups[rest], val_ratio / (1 - test_ratio), seed)
    split[rest[val_rel]] = "val"
    split[rest[train_rel]] = "train"
    return pd.Series(split, index=df.index)


def prepare_out_dir(args) -> Path:
    out_dir = Path(args.out_root) / args.name
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_dir} już istnieje — użyj --overwrite, żeby zbudować go od nowa")
        shutil.rmtree(out_dir)
    return out_dir


def finish(out_dir: Path, df: pd.DataFrame, splits, info: dict, csv_cols: list[str]):
    df[csv_cols].to_csv(out_dir / "split.csv", index=False)
    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(splits, fill_value=0)
    info = {"name": out_dir.name, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), **info,
            "n_images": len(df),
            "counts": {s: {c: int(v) for c, v in counts.loc[s].items()} for s in splits}}
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    print(counts.assign(razem=counts.sum(axis=1), udział=(counts.sum(axis=1) / len(df)).round(3)))
    print(f"zapisano: {out_dir}")


def download(url: str, dst: Path, size: int | None = None):
    if dst.exists() and (size is None or dst.stat().st_size == size):
        return
    print(f"pobieram {dst.name}" + (f" ({size / 1e6:.0f} MB)" if size else ""), flush=True)
    tmp = dst.with_suffix(".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as out:
        shutil.copyfileobj(r, out, length=1 << 20)
    tmp.replace(dst)


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------- train/val/test ----------

def split_and_report(df: pd.DataFrame, args):
    df["group_size"] = df["group"].map(df["group"].value_counts())
    print(f"{df['group'].nunique()} grup, {(df['group_size'] > 1).mean():.1%} obrazów w seriach, "
          f"największa seria: {df['group_size'].max()}")
    df["split"] = assign_train_val_test(df, args.val_ratio, args.test_ratio, args.seed)
    assert df.groupby("group")["split"].nunique().max() == 1, "grupa rozdzielona między zbiory"
    df["path"] = [f"{s}/{label}/{f}" for s, label, f in zip(df["split"], df["label"], df["file"])]


def load_hagrid(args):
    out_dir = prepare_out_dir(args)
    cache = Path(args.cache_dir)
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
                         "source": f"{shard.name}#{i}"})
        print(f"{shard.name}: {len(idx)} obrazów z wybranych gestów", flush=True)
    df = pd.DataFrame(rows)
    print("przed balansowaniem:", df["label"].value_counts().sort_index().to_dict())
    n = min(args.per_class or np.inf, int(df["label"].value_counts().min()))
    df = df.groupby("label").sample(n=n, random_state=args.seed).reset_index(drop=True)
    print(f"po balansowaniu: {n} na klasę, razem {len(df)}")

    thumbs = np.stack([thumbnail(Image.open(io.BytesIO(b))) for b in df["bytes"]])
    df["group"] = groups_from_thumbnails(thumbs, args.group_threshold)
    split_and_report(df, args)
    for data, rel in zip(df["bytes"], df["path"]):
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (out_dir / rel).write_bytes(data)
    finish(out_dir, df, SPLITS, {
        "source": f"hf:{HAGRID_REPO}", "label_map": {label: g for g, label in HAGRID_LABELS.values()},
        "parent": None, "transform": None, "seed": args.seed, "per_class": n, "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio, "group_threshold": args.group_threshold,
        "n_groups": int(df["group"].nunique()),
    }, ["path", "split", "label", "gesture", "group", "group_size", "source"])


def load_kaggle(args):
    out_dir = prepare_out_dir(args)
    archive = Path(args.archive)
    rows = []
    # klasy = podkatalogi z obrazami bezpośrednio w środku; archive/rps-cv-images (kopia) jest pomijane
    for class_dir in sorted(d for d in archive.iterdir() if d.is_dir()):
        files = sorted((f for f in class_dir.iterdir() if f.suffix.lower() in IMG_EXT), key=lambda f: f.name)
        rows += [{"source_path": f.as_posix(), "file": f.name, "label": class_dir.name} for f in files]
    if not rows:
        raise SystemExit(f"Nie znaleziono obrazów w {archive}/<klasa>/ — rozpakuj tam zbiór z Kaggle")
    df = pd.DataFrame(rows)
    print(f"{len(df)} obrazów, klasy: {df['label'].value_counts().sort_index().to_dict()}")
    thumbs = np.stack([thumbnail(Image.open(p)) for p in df["source_path"]])
    df["group"] = groups_from_thumbnails(thumbs, args.group_threshold)
    split_and_report(df, args)
    for src, rel in zip(df["source_path"], df["path"]):
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out_dir / rel)
    finish(out_dir, df, SPLITS, {
        "source": archive.as_posix(), "parent": None, "transform": None, "seed": args.seed,
        "val_ratio": args.val_ratio, "test_ratio": args.test_ratio, "group_threshold": args.group_threshold,
        "n_groups": int(df["group"].nunique()),
    }, ["path", "split", "label", "group", "group_size", "source_path"])


# ---------- zewnętrzne (tylko ewaluacja) ----------

def find_7z() -> str:
    for c in (shutil.which("7z"), r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"):
        if c and Path(c).exists():
            return c
    raise SystemExit("Nie znaleziono 7-Zip (7z) — potrzebny do rozpakowania .rar (https://www.7-zip.org)")


def list_hand_in_wild(cache_dir: Path, classes) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(HIW_RECORD) as r:
        files = {f["key"]: f for f in json.load(r)["files"]}
    rows = []
    for cls in classes:  # pobierane tylko potrzebne archiwa (bez ok.rar, the-finger.rar)
        meta = files[f"{cls}.rar"]
        archive = cache_dir / meta["key"]
        download(meta["links"]["self"], archive, meta["size"])
        if md5_file(archive) != meta["checksum"].removeprefix("md5:"):
            raise SystemExit(f"{archive}: suma MD5 nie zgadza się z Zenodo — usuń plik i pobierz ponownie")
        extract_dir = cache_dir / "extracted"
        if not (extract_dir / cls).is_dir():
            subprocess.run([find_7z(), "x", "-y", f"-o{extract_dir}", str(archive)], check=True,
                           stdout=subprocess.DEVNULL)
        imgs = sorted(f for f in (extract_dir / cls).rglob("*") if f.suffix.lower() in IMG_EXT)
        others = sorted(f.name for f in (extract_dir / cls).rglob("*") if f.is_file() and f.suffix.lower() not in IMG_EXT)
        print(f"{cls}: {len(imgs)} zdjęć, inne pliki (np. adnotacje): {others[:10] or 'brak'}")
        rows += [{"source_path": f, "label": cls, "group_hint": None} for f in imgs]
    return pd.DataFrame(rows)


def list_folder(source: Path, classes) -> pd.DataFrame:
    rows = []
    for cls in classes:
        cls_dir = source / cls
        if not cls_dir.is_dir():
            print(f"brak katalogu {cls_dir} — pomijam klasę {cls}")
            continue
        for f in sorted(cls_dir.rglob("*")):
            if f.suffix.lower() in IMG_EXT:
                rel = f.relative_to(cls_dir)
                rows.append({"source_path": f, "label": cls, "group_hint": rel.parts[0] if len(rel.parts) > 1 else None})
    if not rows:
        raise SystemExit(f"Nie znaleziono zdjęć w {source}/<klasa>/")
    return pd.DataFrame(rows)


def read_photo(path: Path) -> tuple[Image.Image, dict]:
    img = Image.open(path)
    exif = img.getexif()
    taken = exif.get_ifd(ExifTags.IFD.Exif).get(EXIF_DATETIME) or exif.get(306)  # 306 = DateTime
    meta = {"camera": exif.get(272), "exif_orientation": exif.get(274),
            "taken": pd.to_datetime(taken, format="%Y:%m:%d %H:%M:%S", errors="coerce") if taken else pd.NaT}
    img = ImageOps.exif_transpose(img).convert("RGB")  # ImageFolder/PIL nie stosują orientacji z EXIF
    meta["orig_width"], meta["orig_height"] = img.size
    return img, meta


def session_groups(df: pd.DataFrame, gap_min: float) -> pd.Series:
    """Podkatalog osoby/sesji, jeśli jest; inaczej sesje z EXIF: ten sam aparat, przerwa między zdjęciami <= gap_min."""
    groups = pd.Series(index=df.index, dtype=object)
    hinted = df["group_hint"].notna()
    groups[hinted] = "dir:" + df.loc[hinted, "group_hint"].astype(str)
    dated = df[~hinted & df["taken"].notna()].sort_values("taken")
    for camera, part in dated.groupby(dated["camera"].fillna("?")):
        session = (part["taken"].diff() > pd.Timedelta(minutes=gap_min)).cumsum()
        groups[part.index] = [f"exif:{camera}:{s}" for s in session]
    undated = ~hinted & df["taken"].isna()
    groups[undated] = [f"file:{i}" for i in df.index[undated]]
    if undated.any():
        print(f"UWAGA: {undated.sum()} zdjęć bez daty EXIF i bez podkatalogu grupy — każde jest osobną grupą")
    return groups


def contact_sheet(paths: list[Path], out_path: Path, cell: int = 160, cols: int = 12):
    sheet = Image.new("RGB", (cols * cell, max(1, int(np.ceil(len(paths) / cols))) * cell), "white")
    for i, p in enumerate(paths):
        img = Image.open(p)
        img.thumbnail((cell - 4, cell - 4))
        sheet.paste(img, ((i % cols) * cell + 2, (i // cols) * cell + 2))
    sheet.save(out_path, quality=85)


def load_external(args):
    out_dir = prepare_out_dir(args)
    if args.source == "hand-in-wild":
        df = list_hand_in_wild(Path(args.cache_dir), args.classes)
        source, annotations = "zenodo:5503489 Hand_in_Wild (CC BY 4.0)", "brak w archiwach — pełne kadry"
    else:
        df = list_folder(Path(args.dir), args.classes)
        source, annotations = f"folder:{Path(args.dir).resolve().as_posix()}", "brak"

    df["source_md5"] = [md5_file(p) for p in df["source_path"]]
    n0 = len(df)
    df = df.drop_duplicates("source_md5").reset_index(drop=True)
    print(f"{n0} zdjęć, po usunięciu dokładnych duplikatów: {len(df)}")
    images, metas = [], []
    for p in df["source_path"]:
        img, meta = read_photo(p)
        img.thumbnail((args.max_side, args.max_side), Image.LANCZOS)
        images.append(img)
        metas.append(meta)
    df = pd.concat([df, pd.DataFrame(metas)], axis=1)

    overlap = {"checked_against": None}
    train_dir = Path(args.train_dir)
    if train_dir.is_dir():
        files = sorted(f for f in train_dir.rglob("*") if f.suffix.lower() in IMG_EXT)
        print(f"kontrola nakładania: {len(files)} obrazów z {train_dir}", flush=True)
        train_md5 = {hashlib.md5(f.read_bytes()).hexdigest() for f in files}
        sim = normalize(np.stack([thumbnail(im) for im in images])) @ \
            normalize(np.stack([thumbnail(Image.open(f)) for f in files])).T
        df["max_corr_train"] = sim.max(1)
        df["nearest_train"] = [files[i].relative_to(train_dir).as_posix() for i in sim.argmax(1)]
        drop = (df["max_corr_train"] >= args.overlap_threshold) | df["source_md5"].isin(train_md5)
        overlap = {"checked_against": train_dir.as_posix(), "threshold": args.overlap_threshold,
                   "n_dropped": int(drop.sum()), "max_corr": float(df["max_corr_train"].max()),
                   "md5_matches": int(df["source_md5"].isin(train_md5).sum())}
        print(f"nakładanie z {train_dir.name}: max korelacja {overlap['max_corr']:.3f}, "
              f"usuniętych {overlap['n_dropped']} (próg {args.overlap_threshold}, MD5: {overlap['md5_matches']})")
        keep = ~drop.to_numpy()
        df, images = df[keep].reset_index(drop=True), [im for im, k in zip(images, keep) if k]
    else:
        print(f"UWAGA: brak {train_dir} — pominięto kontrolę nakładania z danymi treningowymi")

    df["group"] = session_groups(df, args.session_gap_min)
    n_groups = df["group"].nunique()
    print(f"{n_groups} grup (osoby/sesje): {df['group'].value_counts().to_dict()}")
    if n_groups < 2:
        raise SystemExit("Jedna grupa — nie da się podzielić na external_dev/external_test bez rozdzielania sesji")
    reliable = n_groups >= 4 and df.groupby("label")["group"].nunique().min() >= 2
    if not reliable:
        print("UWAGA: za mało grup na wiarygodny podział — external_dev i external_test mogą mieć te same osoby/sesje")

    _, dev = take_group_fold(df["label"].to_numpy(str), df["group"].to_numpy(), args.dev_ratio, args.seed)
    df["split"] = "external_test"
    df.loc[dev, "split"] = "external_dev"
    assert df.groupby("group")["split"].nunique().max() == 1, "grupa rozdzielona między splity"
    df["path"] = [f"{s}/{label}/{Path(p).stem}_{i:05d}.jpg"
                  for i, (s, label, p) in enumerate(zip(df["split"], df["label"], df["source_path"]))]
    for img, rel in zip(images, df["path"]):
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        img.save(out_dir / rel, quality=95)
    for cls in sorted(df["label"].unique()):
        contact_sheet([out_dir / p for p in df.loc[df["label"] == cls, "path"]], out_dir / f"preview_{cls}.jpg")

    df["source_path"] = [Path(p).as_posix() for p in df["source_path"]]
    finish(out_dir, df, EXTERNAL_SPLITS, {
        "source": source,
        "purpose": "tylko ewaluacja — nie używać w treningu ani do doboru modeli na external_test",
        "classes": sorted(df["label"].unique()), "annotations": annotations, "seed": args.seed,
        "dev_ratio": args.dev_ratio, "max_side": args.max_side,
        "group_method": f"podkatalog osoby/sesji lub sesja EXIF (aparat, przerwa > {args.session_gap_min} min)",
        "n_groups": int(n_groups), "groups_per_split": df.groupby("split")["group"].nunique().to_dict(),
        "group_split_reliable": bool(reliable), "overlap_check": overlap,
    }, ["path", "split", "label", "group", "camera", "taken", "orig_width", "orig_height", "exif_orientation",
        "source_path", "source_md5"] + [c for c in ("max_corr_train", "nearest_train") if c in df])
    print("sprawdź preview_<klasa>.jpg — czy gesty odpowiadają etykietom")


def main():
    args = parse_args()
    {"hagrid": load_hagrid, "kaggle": load_kaggle,
     "hand-in-wild": load_external, "folder": load_external}[args.source](args)


if __name__ == "__main__":
    main()
