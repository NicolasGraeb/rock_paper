"""Green AI: parametry, rozmiar, czas inferencji, pamięć, energia i CO2eq.

    uv run python green.py [--checkpoint ...]
"""
import argparse
import json
import time

import mlflow
import numpy as np
import pandas as pd
import psutil
import torch

from src.checkpoint import DEFAULT_TRACKING_URI, load_checkpoint, results_dir, training_run
from src.models import count_parameters
from train import energy_tracker

MEASURED, ESTIMATED, COMPUTED = "zmierzone", "oszacowane", "obliczone"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None, help="domyślnie najnowszy outputs/baseline/checkpoints/*/best.pt")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=300, help="pomiarów czasu dla batch 1")
    p.add_argument("--throughput-batch", type=int, default=32)
    p.add_argument("--energy-seconds", type=float, default=30, help="czas ciągłej inferencji do pomiaru energii")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    return p.parse_args()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def latency_ms(model, x, device, warmup: int, iters: int) -> np.ndarray:
    for _ in range(warmup):
        model(x)
    sync(device)
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        model(x)
        sync(device)
        times.append((time.perf_counter() - start) * 1000)
    return np.asarray(times)


@torch.no_grad()
def run_for(model, x, device, seconds: float) -> int:
    n, end = 0, time.perf_counter() + seconds
    while time.perf_counter() < end:
        model(x)
        sync(device)
        n += x.shape[0]
    return n


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt, preprocessing, run_id = load_checkpoint(args.checkpoint, device)
    size = preprocessing["image_size"]
    rows = []

    def add(name, value, unit, status, note=""):
        rows.append({"metric": name, "value": value, "unit": unit, "status": status, "note": note})

    add("params_total", count_parameters(model, trainable_only=False), "", COMPUTED)
    add("checkpoint_size_mb", ckpt["path"].stat().st_size / 2**20, "MB", MEASURED, ckpt["path"].name)

    devices = [device] + ([torch.device("cpu")] if device.type == "cuda" else [])
    for dev in devices:
        m = model.to(dev)
        tag = "gpu" if dev.type == "cuda" else "cpu"
        name = torch.cuda.get_device_name(0) if dev.type == "cuda" else f"CPU, {torch.get_num_threads()} wątków"
        t = latency_ms(m, torch.randn(1, 3, size, size, device=dev), dev, args.warmup, args.iters)
        add(f"latency_{tag}_b1_mean_ms", t.mean(), "ms", MEASURED, name)
        add(f"latency_{tag}_b1_median_ms", np.median(t), "ms", MEASURED, name)
        add(f"latency_{tag}_b1_p95_ms", np.percentile(t, 95), "ms", MEASURED, name)
        add(f"fps_{tag}_b1", 1000 / t.mean(), "obrazów/s", COMPUTED, "1 / średni czas, batch 1")
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        xb = torch.randn(args.throughput_batch, 3, size, size, device=dev)
        tb = latency_ms(m, xb, dev, 5, 30)
        add(f"fps_{tag}_b{args.throughput_batch}", args.throughput_batch * 1000 / tb.mean(), "obrazów/s", MEASURED,
            f"batch {args.throughput_batch}")
        if dev.type == "cuda":
            add(f"vram_peak_b{args.throughput_batch}_mb", torch.cuda.max_memory_allocated() / 2**20, "MB", MEASURED,
                "alokator PyTorch (bez narzutu kontekstu CUDA)")
    model.to(device)
    add("ram_process_rss_mb", psutil.Process().memory_info().rss / 2**20, "MB", MEASURED,
        "cały proces Pythona po pomiarach")

    xb = torch.randn(args.throughput_batch, 3, size, size, device=device)
    run_for(model, xb, device, 2)
    tracker = energy_tracker()
    tracker.start()
    n_images = run_for(model, xb, device, args.energy_seconds)
    tracker.stop()
    d = tracker.final_emissions_data
    per_1000 = 1e6 / n_images  #kWh -> Wh/1000 obrazów
    note = f"{n_images} obrazów, batch {args.throughput_batch}, {d.duration:.0f} s, cały komputer"
    add("infer_energy_gpu_wh_per_1000", d.gpu_energy * per_1000, "Wh/1000 obrazów", MEASURED, "NVML; " + note)
    add("infer_energy_cpu_wh_per_1000", d.cpu_energy * per_1000, "Wh/1000 obrazów", MEASURED, "RAPL (Windows EMI); " + note)
    add("infer_energy_ram_wh_per_1000", d.ram_energy * per_1000, "Wh/1000 obrazów", ESTIMATED, "model CodeCarbon; " + note)
    add("infer_energy_total_wh_per_1000", d.energy_consumed * per_1000, "Wh/1000 obrazów", ESTIMATED,
        "suma, zawiera oszacowany RAM; " + note)
    add("infer_co2_g_per_1000", d.emissions * per_1000, "g CO2eq/1000 obrazów", ESTIMATED,
        "współczynnik emisji dla Polski")

    client = mlflow.MlflowClient(args.tracking_uri)
    tm = client.get_run(run_id).data.metrics
    if "train_energy_kwh" in tm:
        add("train_duration_min", tm["train_duration_s"] / 60, "min", MEASURED, "sama pętla treningu")
        add("train_energy_gpu_wh", tm["train_energy_gpu_kwh"] * 1000, "Wh", MEASURED, "NVML")
        add("train_energy_cpu_wh", tm["train_energy_cpu_kwh"] * 1000, "Wh", MEASURED, "RAPL (Windows EMI)")
        add("train_energy_ram_wh", tm["train_energy_ram_kwh"] * 1000, "Wh", ESTIMATED, "model CodeCarbon")
        add("train_energy_total_wh", tm["train_energy_kwh"] * 1000, "Wh", ESTIMATED, "suma, zawiera oszacowany RAM")
        add("train_co2_g", tm["train_co2_kg"] * 1000, "g CO2eq", ESTIMATED, "współczynnik emisji dla Polski")
    else:
        add("train_energy_total_wh", None, "Wh", "nie mierzono", "run bez pomiaru energii")

    out = results_dir(run_id, "green")
    df = pd.DataFrame(rows)
    df.to_csv(out / "green.csv", index=False)
    (out / "green.json").write_text(json.dumps(rows, indent=2, default=float, ensure_ascii=False), encoding="utf-8")
    with training_run(run_id, args.tracking_uri):
        mlflow.log_metrics({f"green_{r['metric']}": float(r["value"]) for r in rows if r["value"] is not None})
        mlflow.log_artifacts(str(out), artifact_path="green")
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"wyniki: {out} | run MLflow: {run_id}")


if __name__ == "__main__":
    main()
