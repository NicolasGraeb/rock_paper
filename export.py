import argparse
import json
from pathlib import Path

import torch

from src.model import build_model
from src.transforms import IMAGENET_MEAN, IMAGENET_STD


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", default="outputs/export")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--android-assets", default="android/app/src/main/assets")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    class_names = ckpt["class_names"]
    image_size = int(ckpt.get("image_size", args.image_size))

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size)
    onnx_path = out_dir / "model.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
        dynamo=False,
    )

    preprocessing = {
        "image_size": image_size,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "class_names": class_names,
        "input_name": "input",
        "output_name": "logits",
    }
    preprocessing_path = out_dir / "preprocessing.json"
    preprocessing_path.write_text(
        json.dumps(preprocessing, indent=2),
        encoding="utf-8",
    )

    assets_dir = Path(args.android_assets)
    assets_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.joinpath("model.onnx").write_bytes(onnx_path.read_bytes())
    assets_dir.joinpath("preprocessing.json").write_text(
        preprocessing_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    print(f"exported={onnx_path}")
    print(f"android_assets={assets_dir}")


if __name__ == "__main__":
    main()
