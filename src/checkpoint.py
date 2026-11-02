from contextlib import contextmanager
from pathlib import Path

import mlflow
import torch

from src.models import build_model
from src.transforms import default_preprocessing

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
DEFAULT_CHECKPOINT_DIR = ROOT / "outputs" / "baseline" / "checkpoints"
RESULTS_DIR = ROOT / "outputs" / "baseline" / "results"


def latest_checkpoint() -> Path:
    found = sorted(DEFAULT_CHECKPOINT_DIR.glob("*/best.pt"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise SystemExit(f"Brak checkpointów w {DEFAULT_CHECKPOINT_DIR}, najpierw uruchom train.py")
    return found[-1]


def load_checkpoint(path: str | Path | None, device: torch.device):
    path = Path(path) if path else latest_checkpoint()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt.get("model_name", "mobilenet_v3_small"), num_classes=len(ckpt["class_names"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    preprocessing = ckpt.get("preprocessing") or default_preprocessing(ckpt.get("image_size", 224))
    ckpt["path"] = path
    return model, ckpt, preprocessing, path.parent.name


def results_dir(run_id: str, name: str) -> Path:
    out = RESULTS_DIR / run_id / name
    out.mkdir(parents=True, exist_ok=True)
    return out


@contextmanager
def training_run(run_id: str, tracking_uri: str = DEFAULT_TRACKING_URI):
    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.MlflowClient().get_run(run_id)
    except mlflow.exceptions.MlflowException:
        raise SystemExit(f"Run {run_id} nie istnieje w {tracking_uri}; checkpoint musi pochodzić z train.py")
    with mlflow.start_run(run_id=run_id) as run:
        yield run
