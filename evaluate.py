"""Ewaluacja checkpointu na HaGRID test.

    uv run python evaluate.py [--checkpoint outputs/baseline/checkpoints/<run_id>/best.pt]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.checkpoint import DEFAULT_TRACKING_URI, ROOT, load_checkpoint, results_dir, training_run
from src.data import create_eval_loader
from src.engine import evaluate_model
from train import classification_metrics_to_log, plot_confusion_matrix


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None, help="domyślnie najnowszy outputs/baseline/checkpoints/*/best.pt")
    p.add_argument("--data-dir", default=str(ROOT / "dataset" / "hagrid_user_split"))
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-errors", type=int, default=24, help="ile błędów (najpewniejszych) pokazać na siatce")
    p.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    return p.parse_args()


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
        img.thumbnail((300, 300))
        ax.imshow(img)
        ax.set_title(f"prawda: {r['true']}\npredykcja: {r['pred']} ({r['confidence']:.0%})", fontsize=8, color="tab:red")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt, preprocessing, run_id = load_checkpoint(args.checkpoint, device)
    class_names = ckpt["class_names"]
    loader = create_eval_loader(args.data_dir, args.split, class_names, preprocessing, batch_size=args.batch_size)

    metrics, y_true, y_pred = evaluate_model(model, loader, device, class_names=class_names, return_probs=True)
    probs = metrics.pop("probs")
    pred_df = pd.DataFrame({
        "path": [p for p, _ in loader.dataset.samples],
        "true": [class_names[i] for i in y_true],
        "pred": [class_names[i] for i in y_pred],
        "confidence": probs.max(1),
        **{f"p_{c}": probs[:, i] for i, c in enumerate(class_names)},
    })
    pred_df["correct"] = pred_df["true"] == pred_df["pred"]

    out = results_dir(run_id, "eval")
    split = args.split
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, out / f"confusion_matrix_{split}.png",
                          title=f"Confusion matrix {split} ({ckpt.get('variant', '')}, epoka {ckpt.get('epoch')})")
    plot_errors(pred_df, class_names, out / f"errors_{split}.png", args.n_errors,
                title=f"Błędne klasyfikacje na {split} (najpewniejsze pierwsze)")
    pred_df.to_csv(out / f"predictions_{split}.csv", index=False)
    pred_df[~pred_df["correct"]].sort_values("confidence", ascending=False).to_csv(
        out / f"errors_{split}.csv", index=False)
    (out / f"classification_report_{split}.json").write_text(
        json.dumps(metrics["classification_report"], indent=2), encoding="utf-8")
    summary = {k: v for k, v in metrics.items() if k != "classification_report"}
    (out / f"metrics_{split}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with training_run(run_id, args.tracking_uri):
        mlflow.log_metrics(classification_metrics_to_log(metrics, split))
        mlflow.log_artifacts(str(out), artifact_path="eval")

    print(json.dumps({k: summary[k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "loss")},
                     indent=2))
    print(f"błędów: {int((~pred_df['correct']).sum())}/{len(pred_df)} | wyniki: {out} | run MLflow: {run_id}")


if __name__ == "__main__":
    main()
