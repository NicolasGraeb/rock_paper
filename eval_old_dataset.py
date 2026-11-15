"""Test wszystkich modeli (od baseline) na całym starym zbiorze dataset_origin (train+val+test), każdy jako nowy run MLflow z Grad-CAM.

    uv run python eval_old_dataset.py
"""
import argparse
import json
import tempfile
import urllib.request
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets

from evaluate import plot_errors
from src.checkpoint import DEFAULT_CHECKPOINT_DIR, DEFAULT_TRACKING_URI, ROOT, load_checkpoint
from src.engine import evaluate_model
from src.explain import grad_cam, plot_side_by_side
from src.transforms import build_transforms
from train import classification_metrics_to_log, energy_metrics, energy_tracker, plot_confusion_matrix
from xai import select

SPLITS = ("train", "val", "test")
SOURCE_EXPERIMENTS = ("papier-kamien-baseline", "papier-kamien-bez-preprocessingu", "papier-kamien-preprocessing")
#parametry runu źródłowego kopiowane do nowego runu
SOURCE_PARAMS = ("augmentation", "training_mode", "normalize", "histogram_equalization", "image_filter",
                 "unfreeze_last", "init_run_id", "lr", "dataset")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(ROOT / "dataset" / "dataset_origin"))
    p.add_argument("--experiment", default="papier-kamien-old-dataset-test")
    p.add_argument("--source-experiments", nargs="+", default=list(SOURCE_EXPERIMENTS),
                   help="eksperymenty, z których brane są modele (runy FINISHED)")
    p.add_argument("--tracking-uri", default="http://127.0.0.1:5000",
                   help="gdy serwer nie odpowiada, zapis prosto do mlflow.db")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--per-class", type=int, default=3, help="przykładów Grad-CAM na klasę (poprawne i błędne osobno)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(ROOT / "outputs" / "old_dataset_test"))
    return p.parse_args()


def tracking_uri(uri: str) -> str:
    if uri.startswith("http"):
        try:
            urllib.request.urlopen(f"{uri}/health", timeout=3)
        except OSError:
            print(f"MLflow pod {uri} nie odpowiada, zapis do {DEFAULT_TRACKING_URI}")
            return DEFAULT_TRACKING_URI
    return uri


def source_runs(experiments) -> pd.DataFrame:
    runs = mlflow.search_runs(experiment_names=list(experiments),
                              filter_string="attributes.status = 'FINISHED'", order_by=["attributes.start_time ASC"])
    exp_names = {e.experiment_id: e.name for e in mlflow.search_experiments()}
    runs["experiment"] = runs["experiment_id"].map(exp_names)
    return runs.reset_index(drop=True)


def old_dataset(data_dir: Path, class_names, preprocessing) -> tuple[ConcatDataset, list[str], list[str]]:
    tf = build_transforms(preprocessing["image_size"], mean=preprocessing["mean"], std=preprocessing["std"],
                          equalize=preprocessing.get("equalize", False),
                          image_filter=preprocessing.get("filter", "none"))
    parts, paths, splits = [], [], []
    for s in SPLITS:
        ds = datasets.ImageFolder(root=str(data_dir / s), transform=tf)
        if ds.classes != list(class_names):
            raise ValueError(f"Klasy w {data_dir / s} ({ds.classes}) różnią się od klas modelu ({class_names})")
        parts.append(ds)
        paths += [p for p, _ in ds.samples]
        splits += [s] * len(ds)
    return ConcatDataset(parts), paths, splits


def run_one(src: pd.Series, args, device) -> dict:
    ckpt_path = DEFAULT_CHECKPOINT_DIR / src["run_id"] / "best.pt"
    model, ckpt, preprocessing, _ = load_checkpoint(ckpt_path, device)
    class_names = ckpt["class_names"]
    ds, paths, splits = old_dataset(Path(args.data_dir), class_names, preprocessing)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")

    tracker = energy_tracker()
    tracker.start()
    metrics, y_true, y_pred = evaluate_model(model, loader, device, class_names=class_names, return_probs=True)
    tracker.stop()
    energy = energy_metrics(tracker, "test")
    energy["test_energy_wh_per_1000_images"] = energy["test_energy_kwh"] * 1e6 / len(ds)
    probs = metrics.pop("probs")
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    pred_df = pd.DataFrame({"path": paths, "old_split": splits,
                            "true": [class_names[i] for i in y_true], "pred": [class_names[i] for i in y_pred],
                            "confidence": probs.max(1), **{f"p_{c}": probs[:, i] for i, c in enumerate(class_names)}})
    pred_df["correct"] = pred_df["true"] == pred_df["pred"]
    acc_by_old_split = pred_df.groupby("old_split")["correct"].mean().to_dict()

    name = src["tags.mlflow.runName"]
    with mlflow.start_run(run_name=name) as run, tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mlflow.set_tags({"source_run_id": src["run_id"], "source_experiment": src["experiment"],
                         "eval_dataset": Path(args.data_dir).name, "image_filter": preprocessing.get("filter", "none")})
        mlflow.log_params({
            "source_run_id": src["run_id"], "source_run_name": name, "source_experiment": src["experiment"],
            **{f"source_{k}": src.get(f"params.{k}") for k in SOURCE_PARAMS},
            "checkpoint": ckpt_path.relative_to(ROOT).as_posix(), "checkpoint_epoch": ckpt.get("epoch"),
            "eval_data_dir": Path(args.data_dir).resolve().relative_to(ROOT).as_posix(), "eval_splits": "train+val+test",
            "eval_size": len(ds), "preprocessing": json.dumps(preprocessing),
        })
        mlflow.log_metrics({**classification_metrics_to_log(metrics, "test"),
                            "test_errors": int((~pred_df["correct"]).sum()),
                            "test_mean_confidence": float(pred_df["confidence"].mean()),
                            **{f"test_accuracy_old_{s}": float(v) for s, v in acc_by_old_split.items()},
                            "hagrid_test_accuracy": float(src["metrics.test_accuracy"]),
                            "hagrid_test_f1_macro": float(src["metrics.test_f1_macro"]), **energy})

        plot_confusion_matrix(metrics["confusion_matrix"], class_names, tmp / "confusion_matrix_old_dataset.png",
                              title=f"{Path(args.data_dir).name}, {name}")
        plot_errors(pred_df, class_names, tmp / "errors_old_dataset.png", 24,
                    title=f"Błędy na {Path(args.data_dir).name}, {name} (najpewniejsze pierwsze)")
        pred_df.to_csv(tmp / "predictions_old_dataset.csv", index=False)
        (tmp / "classification_report_old_dataset.json").write_text(
            json.dumps(metrics["classification_report"], indent=2), encoding="utf-8")
        (tmp / "metrics_old_dataset.json").write_text(
            json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2), encoding="utf-8")
        mlflow.log_artifacts(str(tmp), artifact_path="eval")

        xai_dir = tmp / "xai"
        xai_dir.mkdir()
        size = preprocessing["image_size"]
        records = []
        for kind, correct in (("correct", True), ("errors", False)):
            idx = select(y_true, y_pred, len(class_names), args.per_class, correct, args.seed)
            if not idx:
                continue
            x = torch.stack([ds[i][0] for i in idx]).to(device)
            cams, logits = grad_cam(model, x)
            p = torch.softmax(logits, 1).cpu().numpy()
            rows = []
            for j, i in enumerate(idx):
                pred = int(p[j].argmax())
                rows.append({"image": np.asarray(Image.open(paths[i]).convert("RGB").resize((size, size), Image.BILINEAR)),
                             "true": class_names[y_true[i]], "pred": class_names[pred],
                             "confidence": float(p[j, pred]), "gradcam": cams[j]})
                records.append({"kind": kind, "path": paths[i], "true": class_names[y_true[i]],
                                "pred": class_names[pred], "confidence": float(p[j, pred])})
            label = "poprawne" if correct else "błędne"
            plot_side_by_side(rows, xai_dir / f"gradcam_{kind}.png", f"Grad-CAM, {Path(args.data_dir).name}, {name}: {label}")
        pd.DataFrame(records).to_csv(xai_dir / "selected.csv", index=False)
        mlflow.log_artifacts(str(xai_dir), artifact_path="xai")

        return {"run": name, "eval_run_id": run.info.run_id, "source_run_id": src["run_id"],
                "source_experiment": src["experiment"],
                "augmentacja": src.get("params.augmentation"), "tryb_treningu": src.get("params.training_mode"),
                "przetwarzanie": preprocessing.get("filter", "none") if not preprocessing.get("equalize") else "histeq",
                "normalizacja_imagenet": src.get("params.normalize") != "False",
                "hagrid_test_accuracy": float(src["metrics.test_accuracy"]),
                "old_accuracy": metrics["accuracy"], "old_f1_macro": metrics["f1_macro"],
                "old_precision_macro": metrics["precision_macro"], "old_recall_macro": metrics["recall_macro"],
                "old_loss": metrics["loss"], "old_errors": int((~pred_df["correct"]).sum()),
                **{f"old_f1_{c}": metrics["classification_report"][c]["f1-score"] for c in class_names},
                **{f"old_recall_{c}": metrics["classification_report"][c]["recall"] for c in class_names},
                "old_mean_confidence": float(pred_df["confidence"].mean())}


def summary(experiment: str) -> pd.DataFrame:
    runs = mlflow.search_runs(experiment_names=[experiment], filter_string="attributes.status = 'FINISHED'",
                              order_by=["attributes.start_time ASC"])
    cols = {"tags.mlflow.runName": "run", "run_id": "eval_run_id", "params.source_run_id": "source_run_id",
            "params.source_experiment": "source_experiment", "params.source_augmentation": "augmentacja",
            "params.source_training_mode": "tryb_treningu", "params.source_image_filter": "filtr",
            "params.source_histogram_equalization": "histeq", "params.source_normalize": "normalizacja",
            "metrics.hagrid_test_accuracy": "hagrid_test_accuracy", "metrics.hagrid_test_f1_macro": "hagrid_test_f1_macro",
            "metrics.test_accuracy": "old_accuracy", "metrics.test_f1_macro": "old_f1_macro",
            "metrics.test_precision_macro": "old_precision_macro", "metrics.test_recall_macro": "old_recall_macro",
            "metrics.test_class_paper_f1_score": "old_f1_paper", "metrics.test_class_rock_f1_score": "old_f1_rock",
            "metrics.test_class_scissors_f1_score": "old_f1_scissors", "metrics.test_loss": "old_loss",
            "metrics.test_errors": "old_errors", "metrics.test_mean_confidence": "old_mean_confidence",
            "metrics.test_energy_wh_per_1000_images": "old_energy_wh_per_1000", "metrics.test_co2_kg": "old_co2_kg"}
    df = runs.reindex(columns=list(cols)).rename(columns=cols)
    num = df.select_dtypes("float").columns
    df[num] = df[num].round(6)
    return df


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA niedostępna")
    device = torch.device("cuda")
    mlflow.set_tracking_uri(tracking_uri(args.tracking_uri))
    runs = source_runs(args.source_experiments)
    print(f"{len(runs)} modeli: {list(runs['tags.mlflow.runName'])}")

    mlflow.set_experiment(args.experiment)
    done = summary(args.experiment)["source_run_id"].dropna().tolist()
    for i, src in runs.iterrows():
        if src["run_id"] in done:
            print(f"[{i + 1}/{len(runs)}] {src['tags.mlflow.runName']}: już jest, pomijam")
            continue
        r = run_one(src, args, device)
        print(f"[{i + 1}/{len(runs)}] {r['run']}: acc {r['old_accuracy']:.4f}, f1 {r['old_f1_macro']:.4f}, "
              f"błędów {r['old_errors']} (HaGRID test {r['hagrid_test_accuracy']:.4f})", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary(args.experiment).to_csv(out / "metrics_old_dataset.csv", index=False, encoding="utf-8-sig")
    print(f"zapisano {out / 'metrics_old_dataset.csv'}")


if __name__ == "__main__":
    main()
