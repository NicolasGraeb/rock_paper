"""Kolejka treningów baseline z różnym przetwarzaniem obrazu (--filter), po --jobs naraz, wyniki w MLflow i outputs/comparison_preprocessing.md."""
import argparse
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent

#kolejność według priorytetu, none to punkt odniesienia w tym samym eksperymencie
FILTERS = ["none", "clahe", "gamma", "grayscale", "roi_center", "blur", "white_balance", "sharpen", "canny"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--filters", nargs="+", default=FILTERS)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=3)
    p.add_argument("--experiment", default="papier-kamien-preprocessing")
    p.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    p.add_argument("--log-dir", default=str(ROOT / "outputs" / "logs" / "preprocessing"))
    return p.parse_args()


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def finished_filters(experiment):
    runs = mlflow.search_runs(experiment_names=[experiment], filter_string="attributes.status = 'FINISHED'")
    return set(runs["params.image_filter"]) if len(runs) else set()


def train(name, args):
    log_path = Path(args.log_dir) / f"prep_{name}.log"
    cmd = [sys.executable, "train.py", "--filter", name, "--run-name", f"prep_{name}",
           "--experiment", args.experiment, "--tracking-uri", args.tracking_uri,
           "--num-workers", str(args.num_workers)]
    log(f"start {name}")
    t = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "MLFLOW_DISABLE_AGENT_HINT": "1"}
        code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    log(f"koniec {name}: kod {code}, {(time.time() - t) / 60:.1f} min, log {log_path}")
    return name, code


def write_summary(experiment):
    runs = mlflow.search_runs(experiment_names=[experiment], filter_string="attributes.status = 'FINISHED'",
                              order_by=["metrics.test_f1_macro DESC"])
    lines = ["| filtr | run | val acc | val F1 macro | test acc | test F1 macro | najlepsza epoka | czas treningu [min] |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in runs.iterrows():
        lines.append(f"| {r['params.image_filter']} | {r['run_id'][:8]} | {r['metrics.best_val_accuracy']:.4f} | "
                     f"{r['metrics.best_val_f1_macro']:.4f} | {r['metrics.test_accuracy']:.4f} | "
                     f"{r['metrics.test_f1_macro']:.4f} | {int(r['metrics.best_epoch'])} | "
                     f"{r['metrics.train_duration_s'] / 60:.1f} |")
    out = ROOT / "outputs" / "comparison_preprocessing.md"
    out.write_text(f"# Przetwarzanie obrazu, eksperyment {experiment}\n\n"
                   "MobileNetV3-Small, hagrid_user_split, bez augmentacji, seed 42. Filtr na wszystkich splitach.\n"
                   "Treningi szły równolegle, więc czas i energia z codecarbon są zawyżone.\n\n"
                   + "\n".join(lines) + "\n", encoding="utf-8")
    log(f"podsumowanie: {out}")


def main():
    args = parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlopen(f"{args.tracking_uri}/health", timeout=5)
    except OSError as e:
        raise SystemExit(f"MLflow pod {args.tracking_uri} nie odpowiada ({e}), najpierw uruchom mlflow server")
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    done = finished_filters(args.experiment)
    todo = [f for f in args.filters if f not in done]
    log(f"pominięte (już są w MLflow): {sorted(done & set(args.filters))}, do zrobienia: {todo}, po {args.jobs} naraz")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda f: train(f, args), todo))
    failed = [n for n, code in results if code]
    if failed:
        log(f"nieudane: {failed}")
    write_summary(args.experiment)


if __name__ == "__main__":
    main()
