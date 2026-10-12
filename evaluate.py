import argparse
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay

from src.data import create_dataloaders
from src.engine import evaluate_model
from src.model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="archive/rps-cv-images")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--experiment", default="papier-kamien")
    p.add_argument("--run-name", default="evaluate")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    image_size = ckpt.get("image_size", args.image_size)

    _, val_loader, _ = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    final_metrics, _, _ = evaluate_model(
        model, val_loader, device, class_names=class_names
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "checkpoint": args.checkpoint,
                "data_dir": args.data_dir,
                "batch_size": args.batch_size,
                "image_size": image_size,
                "seed": args.seed,
            }
        )
        mlflow.log_metrics(
            {
                "val_loss": final_metrics["loss"],
                "val_accuracy": final_metrics["accuracy"],
                "val_f1_macro": final_metrics["f1_macro"],
                "val_precision_macro": final_metrics["precision_macro"],
                "val_recall_macro": final_metrics["recall_macro"],
                "val_f1_weighted": final_metrics["f1_weighted"],
                "val_precision_weighted": final_metrics["precision_weighted"],
                "val_recall_weighted": final_metrics["recall_weighted"],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            cm_path = tmp_dir / "confusion_matrix.png"
            report_path = tmp_dir / "classification_report.json"
            fig, ax = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=np.array(final_metrics["confusion_matrix"]),
                display_labels=class_names,
            )
            disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
            ax.set_title("Confusion matrix")
            fig.tight_layout()
            fig.savefig(cm_path, dpi=150)
            plt.close(fig)
            report_path.write_text(
                json.dumps(final_metrics["classification_report"], indent=2),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(report_path))

        printable = {
            k: v
            for k, v in final_metrics.items()
            if k != "classification_report"
        }
        print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
