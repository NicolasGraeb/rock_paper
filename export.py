"""Eksport do ONNX, porównanie z PyTorch i kopia do aplikacji Android.

    uv run python export.py [--checkpoint ...] [--verify-only]
"""
import argparse
import json
import shutil
from pathlib import Path

import mlflow
import numpy as np
import onnxruntime as ort
import torch

from src.checkpoint import DEFAULT_TRACKING_URI, ROOT, load_checkpoint, results_dir, training_run
from src.data import create_eval_loader


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None, help="domyślnie najnowszy outputs/baseline/checkpoints/*/best.pt")
    p.add_argument("--data-dir", default=str(ROOT / "dataset" / "hagrid_user_split"))
    p.add_argument("--n-images", type=int, default=500, help="obrazów z test do weryfikacji (0 = cały test)")
    p.add_argument("--android-assets", default=str(ROOT / "android" / "app" / "src" / "main" / "assets"))
    p.add_argument("--android-test-images", type=int, default=30,
                   help="obrazów z test (z predykcjami PyTorch) do testu zgodności na telefonie, 0 = bez")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    return p.parse_args()


def export_onnx(model, image_size: int, path):
    torch.onnx.export(model, torch.randn(1, 3, image_size, image_size), str(path), input_names=["input"],
                      output_names=["logits"], opset_version=18, dynamo=False,
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})


def verify(model, onnx_path, loader, n_images: int) -> dict:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    pt_logits, ox_logits, targets = [], [], []
    with torch.no_grad():
        for x, y in loader:
            pt_logits.append(model(x).numpy())
            ox_logits.append(session.run(None, {"input": x.numpy()})[0])
            targets.append(y.numpy())
            if n_images and sum(len(t) for t in targets) >= n_images:
                break
    pt, ox, y = (np.concatenate(a)[:n_images or None] for a in (pt_logits, ox_logits, targets))
    softmax = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
    return {
        "n_images": int(len(y)),
        "max_abs_diff_logits": float(np.abs(pt - ox).max()),
        "max_abs_diff_probs": float(np.abs(softmax(pt) - softmax(ox)).max()),
        "prediction_agreement": float((pt.argmax(1) == ox.argmax(1)).mean()),
        "accuracy_pytorch": float((pt.argmax(1) == y).mean()),
        "accuracy_onnx": float((ox.argmax(1) == y).mean()),
    }


def write_android_parity_set(model, loader, class_names, n: int, out_dir: Path):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    idx = sorted(rng.choice(len(loader.dataset), n, replace=False).tolist())
    expected = []
    with torch.no_grad():
        for i in idx:
            x, y = loader.dataset[i]
            probs = torch.softmax(model(x[None]), 1)[0].tolist()
            src = Path(loader.dataset.samples[i][0])
            shutil.copy2(src, out_dir / src.name)
            expected.append({"file": src.name, "true": class_names[y], "probs": probs})
    (out_dir / "expected.json").write_text(json.dumps({"class_names": class_names, "images": expected}, indent=2),
                                           encoding="utf-8")


def main():
    args = parse_args()
    model, ckpt, preprocessing, run_id = load_checkpoint(args.checkpoint, torch.device("cpu"))
    out = results_dir(run_id, "export")
    onnx_path = out / "model.onnx"
    config = {
        "image_size": preprocessing["image_size"],
        "resize": "bilinear, do kwadratu image_size x image_size",
        "color_order": "RGB",
        "scale": "piksel / 255",
        "equalize_luma": preprocessing.get("equalize", False),
        "filter": preprocessing.get("filter", "none"),
        "mean": list(preprocessing["mean"]),
        "std": list(preprocessing["std"]),
        "class_names": ckpt["class_names"],
        "input_name": "input",
        "output_name": "logits",
        "source_run_id": run_id,
    }
    if not args.verify_only:
        export_onnx(model, preprocessing["image_size"], onnx_path)
        (out / "preprocessing.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    if not onnx_path.exists():
        raise SystemExit(f"Brak {onnx_path}, uruchom export.py bez --verify-only")

    loader = create_eval_loader(args.data_dir, "test", ckpt["class_names"], preprocessing)
    result = verify(model, onnx_path, loader, args.n_images)
    result["onnx_size_mb"] = onnx_path.stat().st_size / 2**20
    result["passed"] = result["prediction_agreement"] == 1.0 and result["max_abs_diff_probs"] < 1e-3
    (out / "onnx_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("ONNX niezgodny z PyTorch, nie kopiuję do aplikacji")

    if args.android_assets and not args.verify_only:
        assets = Path(args.android_assets)
        assets.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_path, assets / "model.onnx")
        shutil.copy2(out / "preprocessing.json", assets / "preprocessing.json")
        print(f"skopiowano do {assets}")
        if args.android_test_images:
            parity_dir = assets.parents[1] / "androidTest" / "assets" / "parity"
            write_android_parity_set(model, loader, ckpt["class_names"], args.android_test_images, parity_dir)
            print(f"zestaw do testu na telefonie: {parity_dir}")

    with training_run(run_id, args.tracking_uri):
        mlflow.log_metrics({f"onnx_{k}": float(v) for k, v in result.items()})
        mlflow.log_artifacts(str(out), artifact_path="export")
    print(f"wyniki: {out} | run MLflow: {run_id}")


if __name__ == "__main__":
    main()
