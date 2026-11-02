"""Ewaluacja checkpointu na dowolnym splicie: test HaGRID, external_dev, external_test.

Preprocessing i mapowanie klas brane z checkpointu (to samo co w treningu danego modelu).
Do MLflow trafiają: metryki globalne i per klasa, macierz pomyłek, predykcje per obraz (CSV),
siatka błędnych klasyfikacji z confidence i czas inferencji (ms/obraz).

    uv run python evaluate.py --checkpoint outputs/checkpoints/<run_id>/best.pt --data-dir dataset/external_hiw --split external_dev
    # końcowy test: tylko raz, po wyborze modeli:
    uv run python evaluate.py --checkpoint ... --data-dir dataset/external_hiw --split external_test --final-test
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data import create_eval_loader
from src.engine import evaluate_model
from src.models import build_model, count_parameters
from src.transforms import default_preprocessing
from train import (
    DEFAULT_TRACKING_URI,
    ROOT,
    classification_metrics_to_log,
    load_dataset_info,
    plot_confusion_matrix,
)

# osobne eksperymenty: diagnostyka na external_dev, nietknięty końcowy test na external_test
DEFAULT_EXPERIMENTS = {"external_dev": "papier-kamien-external-dev", "external_test": "papier-kamien-external-test"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default=str(ROOT / "dataset" / "dataset_origin_new"))
    p.add_argument("--split", default="test", help="podkatalog data-dir: test, val, external_dev, external_test, ...")
    p.add_argument("--final-test", action="store_true",
                   help="wymagane dla --split external_test (zabezpieczenie przed użyciem testu przy doborze modeli)")
    p.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    p.add_argument("--experiment", default=None,
                   help="domyślnie papier-kamien-external-dev / -external-test dla splitów zewnętrznych, "
                        "papier-kamien-eval dla pozostałych")
    p.add_argument("--run-name", default=None, help="domyślnie <wariant z checkpointu>_<split>")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--exclude-csv", default=None,
                   help="CSV z kolumną path (względem data-dir), obrazy pominięte, np. niezgodne z konwencją gestu")
    p.add_argument("--subset-name", default=None, help="nazwa podzbioru (tag/sufiks runu), domyślnie nazwa pliku CSV")
    p.add_argument("--n-errors", type=int, default=24, help="ile błędnych predykcji pokazać na siatce")
    p.add_argument("--benchmark-batch-size", type=int, default=32)
    p.add_argument("--benchmark-iters", type=int, default=50)
    p.add_argument("--no-benchmark", action="store_true")
    args = p.parse_args()
    if args.split == "external_test" and not args.final_test:
        p.error("external_test to końcowy test, uruchamiaj go dopiero po wyborze modeli, z flagą --final-test")
    if args.exclude_csv and args.subset_name is None:
        args.subset_name = Path(args.exclude_csv).stem
    if args.experiment is None:
        args.experiment = DEFAULT_EXPERIMENTS.get(args.split, "papier-kamien-eval")
    return args


@torch.no_grad()
def benchmark_ms_per_image(model, device, image_size: int, batch_size: int, iters: int, warmup: int = 10) -> float:
    """Czas samego forward pass (bez wczytywania obrazów) na losowym wejściu, ms na obraz.

    Stałe: urządzenie, rozdzielczość, batch size; warm-up i synchronizacja CUDA przed/po pomiarze.
    """
    model.eval()
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return float(np.median(times) * 1000 / batch_size)


def image_stats(path: str) -> dict:
    img = Image.open(path)
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    return {"width": img.width, "height": img.height, "brightness": float(gray.mean()), "contrast": float(gray.std())}


def plot_errors(pred_df: pd.DataFrame, class_names, out_path: Path, n: int, title: str):
    errors = pred_df[~pred_df["correct"]].sort_values("confidence", ascending=False).head(n)
    if errors.empty:
        return False
    cols = min(6, len(errors))
    rows = int(np.ceil(len(errors) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), errors.iterrows()):
        img = Image.open(r["path"])
        img.thumbnail((400, 400))
        ax.imshow(img)
        probs = " ".join(f"{c[:3]}={r[f'p_{c}']:.2f}" for c in class_names)
        ax.set_title(f"true: {r['true']} → pred: {r['pred']}\n{probs}", fontsize=8, color="tab:red")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    preprocessing = ckpt.get("preprocessing") or default_preprocessing(ckpt.get("image_size", 224))
    model_name = ckpt.get("model_name", "mobilenet_v3_small")  # starsze checkpointy nie miały nazwy
    variant = ckpt.get("variant", model_name)
    source_run_id = ckpt_path.parent.name  # checkpointy zapisywane w outputs/checkpoints/<run_id>/
    data_dir = Path(args.data_dir)
    dataset_info = load_dataset_info(data_dir)
    dataset_name = dataset_info.get("name", data_dir.name)

    exclude = set(pd.read_csv(args.exclude_csv)["path"]) if args.exclude_csv else set()
    loader = create_eval_loader(data_dir, args.split, class_names, preprocessing,
                                batch_size=args.batch_size, num_workers=args.num_workers, exclude=exclude)
    n_excluded = sum(p.startswith(f"{args.split}/") for p in exclude)

    model = build_model(model_name, num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    metrics, y_true, y_pred = evaluate_model(model, loader, device, class_names=class_names, return_probs=True)
    probs = metrics.pop("probs")
    ms_per_image = None
    if not args.no_benchmark:
        ms_per_image = benchmark_ms_per_image(model, device, preprocessing["image_size"],
                                              args.benchmark_batch_size, args.benchmark_iters)

    paths = [p for p, _ in loader.dataset.samples]
    pred_df = pd.DataFrame({
        "path": paths,
        "true": [class_names[i] for i in y_true],
        "pred": [class_names[i] for i in y_pred],
        **{f"p_{c}": probs[:, i] for i, c in enumerate(class_names)},
    })
    pred_df["confidence"] = probs.max(1)
    pred_df["correct"] = pred_df["true"] == pred_df["pred"]
    pred_df = pd.concat([pred_df, pd.DataFrame([image_stats(p) for p in paths])], axis=1)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    subset = args.subset_name or "all"
    run_name = args.run_name or f"{variant}_{args.split}" + ("" if subset == "all" else f"_{subset}")

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"dataset": dataset_name, "model": model_name, "variant": variant,
                         "eval_split": args.split, "source_run_id": source_run_id, "subset": subset,
                         "training_mode": ckpt.get("training_mode", "full")})
        mlflow.log_params(
            {
                "checkpoint": ckpt_path.as_posix(),
                "checkpoint_epoch": ckpt.get("epoch"),
                "source_run_id": source_run_id,
                "data_dir": data_dir.as_posix(),
                "dataset": dataset_name,
                "dataset_created": dataset_info.get("created"),
                "split": args.split,
                "n_images": len(pred_df),
                "exclude_csv": args.exclude_csv,
                "n_excluded": n_excluded,
                "classes_present": ",".join(sorted(pred_df["true"].unique())),
                "label_map": json.dumps({c: i for i, c in enumerate(class_names)}),
                "preprocessing": json.dumps(preprocessing),
                "n_params": count_parameters(model, trainable_only=False),
                "batch_size": args.batch_size,
                "benchmark_batch_size": args.benchmark_batch_size,
                "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            }
        )
        to_log = classification_metrics_to_log(metrics, args.split)
        if ms_per_image is not None:
            to_log["ms_per_image"] = ms_per_image
        mlflow.log_metrics(to_log)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            cm_path = tmp_dir / f"confusion_matrix_{args.split}.png"
            plot_confusion_matrix(metrics["confusion_matrix"], class_names, cm_path,
                                  title=f"{variant} - {args.split}")
            (tmp_dir / f"classification_report_{args.split}.json").write_text(
                json.dumps(metrics["classification_report"], indent=2), encoding="utf-8")
            (tmp_dir / "metrics.json").write_text(json.dumps(
                {k: v for k, v in metrics.items() if k != "classification_report"} | {"ms_per_image": ms_per_image},
                indent=2), encoding="utf-8")
            pred_df.to_csv(tmp_dir / f"predictions_{args.split}.csv", index=False)
            plot_errors(pred_df, class_names, tmp_dir / f"errors_{args.split}.png", args.n_errors,
                        title=f"{variant}: błędne klasyfikacje na {args.split} (najpewniejsze pierwsze)")
            mlflow.log_artifacts(str(tmp_dir))

        printable = {k: v for k, v in metrics.items() if k != "classification_report"}
        print(json.dumps(printable | {"ms_per_image": ms_per_image}, indent=2))
        n_err = int((~pred_df["correct"]).sum())
        print(f"błędów: {n_err}/{len(pred_df)} | eksperyment MLflow: {args.experiment}, run: {run_name}")


if __name__ == "__main__":
    main()
