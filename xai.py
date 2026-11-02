"""Grad-CAM i occlusion sensitivity na HaGRID test.

    uv run python xai.py [--checkpoint ...] [--per-class 3]
"""
import argparse

import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.checkpoint import DEFAULT_TRACKING_URI, ROOT, load_checkpoint, results_dir, training_run
from src.data import create_eval_loader
from src.engine import evaluate_model
from src.explain import grad_cam, occlusion_sensitivity, plot_side_by_side


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None, help="domyślnie najnowszy outputs/baseline/checkpoints/*/best.pt")
    p.add_argument("--data-dir", default=str(ROOT / "dataset" / "hagrid_user_split"))
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--per-class", type=int, default=3, help="przykładów na klasę (poprawnych i błędnych osobno)")
    p.add_argument("--patch", type=int, default=32, help="bok zasłony w occlusion sensitivity (px)")
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    return p.parse_args()


def select(y_true, y_pred, n_classes: int, per_class: int, correct: bool, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(n_classes):
        pool = np.flatnonzero((y_true == c) & ((y_pred == y_true) == correct))
        idx += sorted(rng.choice(pool, min(per_class, len(pool)), replace=False).tolist())
    return idx


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt, preprocessing, run_id = load_checkpoint(args.checkpoint, device)
    class_names = ckpt["class_names"]
    loader = create_eval_loader(args.data_dir, args.split, class_names, preprocessing)
    _, y_true, y_pred = evaluate_model(model, loader, device, class_names=class_names)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    size = preprocessing["image_size"]
    out = results_dir(run_id, "xai")

    records = []
    for kind, correct in (("correct", True), ("errors", False)):
        idx = select(y_true, y_pred, len(class_names), args.per_class, correct, args.seed)
        if not idx:
            print(f"brak przykładów: {kind}")
            continue
        x = torch.stack([loader.dataset[i][0] for i in idx]).to(device)
        cams, logits = grad_cam(model, x)
        probs = torch.softmax(logits, 1).cpu().numpy()
        rows = []
        for j, i in enumerate(idx):
            path = loader.dataset.samples[i][0]
            pred = int(probs[j].argmax())
            occ = occlusion_sensitivity(model, x[j:j + 1], int(y_true[i]), args.patch, args.stride)
            rows.append({"image": np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)),
                         "true": class_names[y_true[i]], "pred": class_names[pred], "confidence": float(probs[j, pred]),
                         "gradcam": cams[j], "occlusion": occ})
            records.append({"kind": kind, "path": path, "true": class_names[y_true[i]], "pred": class_names[pred],
                            "confidence": float(probs[j, pred]),
                            "occlusion_max_drop": float(occ.max()), "occlusion_min_drop": float(occ.min())})
        label = "poprawne" if correct else "błędne"
        plot_side_by_side(rows, out / f"gradcam_{kind}.png", f"Grad-CAM, {args.split}: {label} predykcje")
        plot_side_by_side(rows, out / f"occlusion_{kind}.png",
                          f"Grad-CAM i occlusion sensitivity ({args.patch} px, krok {args.stride}), {args.split}: {label}",
                          maps=("gradcam", "occlusion"))
    pd.DataFrame(records).to_csv(out / "selected.csv", index=False)

    with training_run(run_id, args.tracking_uri):
        mlflow.log_artifacts(str(out), artifact_path="xai")
    print(f"zapisano {len(records)} przykładów: {out} | run MLflow: {run_id}")


if __name__ == "__main__":
    main()
