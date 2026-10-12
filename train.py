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
from src.engine import train_model
from src.model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="archive/rps-cv-images")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--experiment", default="papier-kamien")
    p.add_argument("--run-name", default="baseline")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_learning_curves(history, out_path: Path):
    epochs = range(1, len(history.train_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history.train_loss, label="train")
    axes[0].plot(epochs, history.val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history.train_acc, label="train")
    axes[1].plot(epochs, history.val_acc, label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, val_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    model = build_model(num_classes=len(class_names))

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "data_dir": args.data_dir,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "image_size": args.image_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "val_ratio": args.val_ratio,
                "seed": args.seed,
                "num_classes": len(class_names),
                "class_names": ",".join(class_names),
                "model": "SmallCNN",
                "optimizer": "AdamW",
                "device": str(device),
                "train_size": len(train_loader.dataset),
                "val_size": len(val_loader.dataset),
            }
        )

        def on_epoch(epoch, train_metrics, val_metrics):
            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "train_f1_macro": train_metrics["f1_macro"],
                    "train_precision_macro": train_metrics["precision_macro"],
                    "train_recall_macro": train_metrics["recall_macro"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1_macro": val_metrics["f1_macro"],
                    "val_precision_macro": val_metrics["precision_macro"],
                    "val_recall_macro": val_metrics["recall_macro"],
                    "val_f1_weighted": val_metrics["f1_weighted"],
                    "val_precision_weighted": val_metrics["precision_weighted"],
                    "val_recall_weighted": val_metrics["recall_weighted"],
                },
                step=epoch,
            )
            print(
                f"epoch {epoch:03d} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1={val_metrics['f1_macro']:.4f}"
            )

        history, final_metrics, _, _ = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            class_names=class_names,
            on_epoch=on_epoch,
        )

        mlflow.log_metrics(
            {
                "best_val_accuracy": final_metrics["accuracy"],
                "best_val_f1_macro": final_metrics["f1_macro"],
                "best_val_precision_macro": final_metrics["precision_macro"],
                "best_val_recall_macro": final_metrics["recall_macro"],
                "best_val_f1_weighted": final_metrics["f1_weighted"],
                "best_val_precision_weighted": final_metrics["precision_weighted"],
                "best_val_recall_weighted": final_metrics["recall_weighted"],
            }
        )

        for name, values in final_metrics["classification_report"].items():
            if not isinstance(values, dict):
                continue
            safe = name.replace(" ", "_")
            for metric_name in ("precision", "recall", "f1-score", "support"):
                if metric_name in values:
                    key = f"class_{safe}_{metric_name.replace('-', '_')}"
                    mlflow.log_metric(key, float(values[metric_name]))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            curves_path = tmp_dir / "learning_curves.png"
            cm_path = tmp_dir / "confusion_matrix.png"
            report_path = tmp_dir / "classification_report.json"
            metrics_path = tmp_dir / "metrics.json"
            model_path = tmp_dir / "model.pt"
            history_path = tmp_dir / "history.json"

            plot_learning_curves(history, curves_path)
            plot_confusion_matrix(
                final_metrics["confusion_matrix"], class_names, cm_path
            )

            report_path.write_text(
                json.dumps(final_metrics["classification_report"], indent=2),
                encoding="utf-8",
            )
            metrics_path.write_text(
                json.dumps(
                    {
                        k: v
                        for k, v in final_metrics.items()
                        if k != "classification_report"
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            history_path.write_text(
                json.dumps(
                    {
                        "train_loss": history.train_loss,
                        "val_loss": history.val_loss,
                        "train_acc": history.train_acc,
                        "val_acc": history.val_acc,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": args.image_size,
                    "model_name": "SmallCNN",
                },
                model_path,
            )

            mlflow.log_artifact(str(curves_path))
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(report_path))
            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(history_path))
            mlflow.log_artifact(str(model_path))

        print(f"run_id={mlflow.active_run().info.run_id}")
        print(f"val_accuracy={final_metrics['accuracy']:.4f}")
        print(f"val_f1_macro={final_metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
