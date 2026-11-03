import argparse
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from codecarbon import OfflineEmissionsTracker
from tqdm.auto import tqdm

from src.data import create_dataloaders
from src.engine import MONITORS, evaluate_model, train_model
from src.models import MODELS, build_model, count_parameters, get_spec, set_trainable

ROOT = Path(__file__).resolve().parent
DEFAULT_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(ROOT / "dataset" / "hagrid_user_split"),
                   help="wariant datasetu z podkatalogami train/val/test (patrz load_dataset.py)")
    p.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    p.add_argument("--experiment", default="papier-kamien-baseline")
    p.add_argument("--model", choices=list(MODELS), default="mobilenet_v3_small",
                   help="architektura z rejestru src/models (MODELS)")
    p.add_argument("--run-name", default=None, help="domyślnie nazwa modelu")
    p.add_argument("--epochs", type=int, default=50, help="maksymalna liczba epok (early stopping może skończyć wcześniej)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=None, help="domyślnie default_lr modelu z rejestru")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None,
                   help="domyślnie: wagi ImageNet, jeśli model je ma")
    p.add_argument("--monitor", choices=list(MONITORS), default="val_loss",
                   help="metryka walidacyjna wybierająca najlepszą epokę i sterująca early stoppingiem")
    p.add_argument("--patience", type=int, default=5,
                   help="early stopping po tylu epokach bez poprawy; 0 = wyłączony")
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--checkpoint-dir", default=str(ROOT / "outputs" / "baseline" / "checkpoints"),
                   help="best.pt i last.pt zapisywane w <checkpoint-dir>/<run_id>/")
    p.add_argument("--log-every-n-steps", type=int, default=10)
    p.add_argument("--freeze-features", action="store_true",
                   help="zamroź ekstraktor cech (.features), trenuj tylko .classifier (feature extraction)")
    p.add_argument("--unfreeze-last", type=int, default=0,
                   help="przy --freeze-features odmroź ostatnie N bloków .features (fine-tuning)")
    p.add_argument("--init-checkpoint", default=None,
                   help="start z wag checkpointu (np. best.pt wariantu feature extraction) zamiast z ImageNet")
    p.add_argument("--no-normalize", action="store_true",
                   help="bez normalizacji mean/std, model dostaje piksele 0-1 (resize zostaje)")
    p.add_argument("--equalize", action="store_true",
                   help="wyrównanie histogramu jasności (kanał Y) po resize, na wszystkich splitach")
    p.add_argument("--bn-eval", action="store_true",
                   help="wszystkie BatchNorm w trybie eval podczas treningu (stałe statystyki); "
                        "BN zamrożonych warstw jest w eval zawsze")
    args = p.parse_args()

    spec = get_spec(args.model)
    if args.pretrained is None:
        args.pretrained = spec.supports_pretrained
    if args.lr is None:
        args.lr = spec.default_lr
    if args.run_name is None:
        args.run_name = args.model
    if args.unfreeze_last and not args.freeze_features:
        p.error("--unfreeze-last wymaga --freeze-features")
    return args


def training_mode(args) -> str:
    if not args.freeze_features:
        return "full" if not args.pretrained else "full_finetune"
    return "fine_tuning" if args.unfreeze_last else "feature_extraction"


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA niedostępna, trening uruchamiany jest tylko na GPU. "
            f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}. "
            "Sprawdź sterownik NVIDIA i czy zainstalowany jest torch z indeksu cu128 (uv sync)."
        )
    return torch.device("cuda")


def load_dataset_info(data_dir: Path) -> dict:
    path = data_dir / "dataset_info.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def classification_metrics_to_log(metrics: dict, prefix: str) -> dict:
    keys = ("loss", "accuracy", "f1_macro", "precision_macro", "recall_macro",
            "f1_weighted", "precision_weighted", "recall_weighted")
    out = {f"{prefix}_{k}": metrics[k] for k in keys}
    for name, values in metrics["classification_report"].items():
        if not isinstance(values, dict) or name in ("macro avg", "weighted avg"):
            continue
        for metric_name in ("precision", "recall", "f1-score"):
            out[f"{prefix}_class_{name}_{metric_name.replace('-', '_')}"] = float(values[metric_name])
    return out


def energy_tracker() -> OfflineEmissionsTracker:
    #GPU z NVML, CPU z RAPL, RAM i CO2 szacowane
    return OfflineEmissionsTracker(country_iso_code="POL", measure_power_secs=1, save_to_file=False,
                                   log_level="error", gpu_ids=[0])


def energy_metrics(tracker: OfflineEmissionsTracker, prefix: str) -> dict:
    d = tracker.final_emissions_data
    return {f"{prefix}_energy_kwh": d.energy_consumed, f"{prefix}_energy_gpu_kwh": d.gpu_energy,
            f"{prefix}_energy_cpu_kwh": d.cpu_energy, f"{prefix}_energy_ram_kwh": d.ram_energy,
            f"{prefix}_co2_kg": d.emissions, f"{prefix}_duration_s": d.duration}


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_learning_curves(history, out_path: Path, batch_losses=None, steps_per_epoch=None):
    epochs = range(1, len(history.train_loss) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    if batch_losses and steps_per_epoch:
        x = np.arange(1, len(batch_losses) + 1) / steps_per_epoch
        axes[0].plot(x, batch_losses, color="tab:blue", alpha=0.25, lw=0.8, label="train (batch)")
    axes[0].plot(epochs, history.train_loss, "o-", color="tab:blue", label="train (średnia z epoki)")
    axes[0].plot(epochs, history.val_loss, "o-", color="tab:orange", label="val")
    axes[0].set_title("Loss")
    axes[0].set_ylabel("loss (skala log)")
    axes[0].set_yscale("log")

    panels = [
        (axes[1], "Accuracy", history.train_acc, history.val_acc),
        (axes[2], "F1 macro", history.train_f1, history.val_f1),
    ]
    for ax, title, train_values, val_values in panels:
        ax.plot(epochs, train_values, "o-", color="tab:blue", label="train")
        ax.plot(epochs, val_values, "o-", color="tab:orange", label="val")
        ax.set_title(title)
        ax.set_ylabel(title.lower())

    for ax in axes:
        if history.best_epoch:
            ax.axvline(history.best_epoch, color="tab:green", ls="--", lw=1, label=f"best (epoka {history.best_epoch})")
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, out_path: Path, title: str = "Confusion matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    device = require_cuda()
    set_seed(args.seed)
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    data_dir = Path(args.data_dir)
    dataset_info = load_dataset_info(data_dir)
    dataset_name = dataset_info.get("name", data_dir.name)
    spec = get_spec(args.model)
    mean, std = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) if args.no_normalize else (spec.mean, spec.std)
    preprocessing = {"image_size": args.image_size, "resize": "square", "mean": list(mean), "std": list(std),
                     "equalize": args.equalize}
    loaders, class_names = create_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        num_workers=args.num_workers,
        mean=mean,
        std=std,
        equalize=args.equalize,
    )
    train_loader, val_loader, test_loader = loaders["train"], loaders["val"], loaders["test"]
    model = build_model(args.model, num_classes=len(class_names), pretrained=args.pretrained)
    init_info = {}
    if args.init_checkpoint:
        init = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if init.get("model_name") != args.model or init.get("class_names") != class_names:
            raise SystemExit(f"--init-checkpoint: model {init.get('model_name')}/{init.get('class_names')} "
                             f"!= {args.model}/{class_names}")
        init_prep = {"equalize": False, **init.get("preprocessing", preprocessing)}
        if init_prep != preprocessing:
            raise SystemExit(f"--init-checkpoint ma inny preprocessing: {init_prep} != {preprocessing}")
        model.load_state_dict(init["model_state_dict"])
        init_info = {"init_checkpoint": Path(args.init_checkpoint).as_posix(),
                     "init_run_id": Path(args.init_checkpoint).parent.name, "init_epoch": init.get("epoch")}
    set_trainable(model, args.freeze_features, args.unfreeze_last)
    n_params = count_parameters(model, trainable_only=False)
    n_trainable = count_parameters(model)
    mode = training_mode(args)
    print(f"model: {args.model} (pretrained={args.pretrained}, {n_params / 1e6:.2f} M parametrów, "
          f"trenowanych {n_trainable / 1e6:.3f} M, tryb {mode}), lr={args.lr}")
    patience = args.patience or None

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tags({"dataset": dataset_name, "model": args.model, "training_mode": mode,
                         "variant": args.run_name})
        mlflow.log_params(
            {
                "data_dir": data_dir.as_posix(),
                "dataset": dataset_name,
                "dataset_transform": dataset_info.get("transform"),
                "split_seed": dataset_info.get("seed"),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "image_size": args.image_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "num_classes": len(class_names),
                "class_names": ",".join(class_names),
                "model": args.model,
                "n_params": n_params,
                "n_params_trainable": n_trainable,
                "training_mode": mode,
                "freeze_features": args.freeze_features,
                "unfreeze_last": args.unfreeze_last,
                "bn_eval": args.bn_eval,
                **init_info,
                "normalize": not args.no_normalize,
                "norm_mean": ",".join(f"{v:.3f}" for v in mean),
                "norm_std": ",".join(f"{v:.3f}" for v in std),
                "label_map": json.dumps({c: i for i, c in enumerate(class_names)}),
                "dataset_created": dataset_info.get("created"),
                "dataset_source": dataset_info.get("source"),
                "variant": args.run_name,
                "pretrained": args.pretrained,
                "optimizer": "AdamW",
                "device": torch.cuda.get_device_name(0),
                "train_size": len(train_loader.dataset),
                "val_size": len(val_loader.dataset),
                "test_size": len(test_loader.dataset),
                "monitor": args.monitor,
                "patience": patience,
                "min_delta": args.min_delta,
                "augmentation": dataset_info.get("transform") or "none",
                "histogram_equalization": args.equalize,
            }
        )
        for name in ("dataset_info.json", "split.csv"):
            if (data_dir / name).exists():
                mlflow.log_artifact(str(data_dir / name), artifact_path="dataset")
        run_id = mlflow.active_run().info.run_id
        checkpoint_dir = Path(args.checkpoint_dir) / run_id
        checkpoint_extra = {
            "class_names": class_names,
            "image_size": args.image_size,
            "model_name": args.model,
            "variant": args.run_name,
            "pretrained": args.pretrained,
            "preprocessing": preprocessing,
            "training_mode": mode,
            "dataset": dataset_name,
            "seed": args.seed,
            "augmentation": dataset_info.get("transform") or "none",
        }

        step = 0
        step_losses = []
        batch_losses = []
        steps_per_epoch = len(train_loader)
        live_curves_path = checkpoint_dir / "learning_curves.png"

        def on_batch(loss):
            nonlocal step
            step += 1
            step_losses.append(loss)
            batch_losses.append(loss)
            if step % args.log_every_n_steps == 0:
                mlflow.log_metric("train_loss_step", float(np.mean(step_losses)), step=step)
                step_losses.clear()

        def on_epoch(epoch, train_metrics, val_metrics, history):
            plot_learning_curves(history, live_curves_path, batch_losses, steps_per_epoch)
            mlflow.log_artifact(str(live_curves_path))
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
            tqdm.write(
                f"epoch {epoch:03d} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1={val_metrics['f1_macro']:.4f}"
            )

        tracker = energy_tracker()
        tracker.start()
        history, val_metrics, _, _ = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            class_names=class_names,
            on_epoch=on_epoch,
            on_batch=on_batch,
            monitor=args.monitor,
            patience=patience,
            min_delta=args.min_delta,
            checkpoint_dir=checkpoint_dir,
            checkpoint_extra=checkpoint_extra,
            bn_eval=args.bn_eval,
        )
        tracker.stop()
        energy = energy_metrics(tracker, "train")
        tracker = energy_tracker()
        tracker.start()
        test_metrics, _, _ = evaluate_model(model, test_loader, device, class_names=class_names)
        tracker.stop()
        energy.update(energy_metrics(tracker, "test"))
        energy["test_energy_wh_per_1000_images"] = energy["test_energy_kwh"] * 1e6 / len(test_loader.dataset)
        mlflow.log_metrics(energy)

        mlflow.log_metrics(
            {
                **classification_metrics_to_log(val_metrics, "best_val"),
                **classification_metrics_to_log(test_metrics, "test"),
                "best_epoch": history.best_epoch,
                "epochs_trained": len(history.train_loss),
            }
        )
        mlflow.set_tag("stopped_early", history.stopped_early)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            curves_path = tmp_dir / "learning_curves.png"
            metrics_path = tmp_dir / "metrics.json"
            model_path = tmp_dir / "model.pt"
            history_path = tmp_dir / "history.json"

            plot_learning_curves(history, curves_path, batch_losses, steps_per_epoch)
            (tmp_dir / "energy.json").write_text(json.dumps(energy, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(tmp_dir / "energy.json"))
            for split, metrics in (("val", val_metrics), ("test", test_metrics)):
                cm_path = tmp_dir / f"confusion_matrix_{split}.png"
                report_path = tmp_dir / f"classification_report_{split}.json"
                plot_confusion_matrix(
                    metrics["confusion_matrix"], class_names, cm_path,
                    title=f"Confusion matrix {split} (epoka {history.best_epoch})",
                )
                report_path.write_text(
                    json.dumps(metrics["classification_report"], indent=2),
                    encoding="utf-8",
                )
                mlflow.log_artifact(str(cm_path))
                mlflow.log_artifact(str(report_path))

            metrics_path.write_text(
                json.dumps(
                    {
                        split: {k: v for k, v in metrics.items() if k != "classification_report"}
                        for split, metrics in (("val", val_metrics), ("test", test_metrics))
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
                        "train_f1_macro": history.train_f1,
                        "val_f1_macro": history.val_f1,
                        "best_epoch": history.best_epoch,
                        "batch_losses": batch_losses,
                        "steps_per_epoch": steps_per_epoch,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            torch.save(
                {
                    **checkpoint_extra,
                    "model_state_dict": model.state_dict(),
                    "epoch": history.best_epoch,
                },
                model_path,
            )

            mlflow.log_artifact(str(curves_path))
            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(history_path))
            mlflow.log_artifact(str(model_path))

        print(f"run_id={run_id}")
        print(f"best_epoch={history.best_epoch} ({args.monitor}), checkpoints={checkpoint_dir}")
        for split, metrics in (("val", val_metrics), ("test", test_metrics)):
            print(f"{split}: accuracy={metrics['accuracy']:.4f} f1_macro={metrics['f1_macro']:.4f} loss={metrics['loss']:.4f}")
        print(f"energia: trening {energy['train_energy_kwh'] * 1000:.1f} Wh ({energy['train_duration_s'] / 60:.1f} min), "
              f"test {energy['test_energy_wh_per_1000_images']:.3f} Wh / 1000 obrazów, CO2 trening {energy['train_co2_kg'] * 1000:.1f} g")


if __name__ == "__main__":
    main()
