"""Cały eksperyment z augmentacją offline z pełnymi obrotami i odbiciami, wszystko po kolei (rzetelne CO2).

Zbiór -> treningi -> evaluate, xai (Grad-CAM), green dla każdego modelu -> test na starym zbiorze (poziomo i pionowo)
-> podsumowanie w outputs/augrot/. Po przerwaniu wystarczy uruchomić ponownie, zrobione kroki są pomijane.

    uv run python run_augrot.py
"""
import argparse
import ctypes
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "hagrid_user_split_aug_rot"
CKPT = ROOT / "outputs" / "baseline" / "checkpoints"
OUT = ROOT / "outputs" / "augrot"
LOGS = ROOT / "outputs" / "logs" / "augrot"
EXPERIMENT = "papier-kamien-aug-rotacje"
OLD_TESTS = {
    "poziomo": ("dataset_origin", "papier-kamien-augrot-old-dataset-test"),
    "pionowo": ("dataset_origin_rot90", "papier-kamien-augrot-old-dataset-rot90-test"),
}
#te same ustawienia co w poprzednich eksperymentach, inny zbiór treningowy
CONFIGS = [
    ("augrot_baseline", []),
    ("augrot_bez_normalizacji", ["--no-normalize"]),
    ("augrot_histeq", ["--equalize"]),
    ("augrot_clahe", ["--filter", "clahe"]),
    ("augrot_gamma", ["--filter", "gamma"]),
    ("augrot_grayscale", ["--filter", "grayscale"]),
    ("augrot_roi_center", ["--filter", "roi_center"]),
    ("augrot_blur", ["--filter", "blur"]),
    ("augrot_white_balance", ["--filter", "white_balance"]),
    ("augrot_sharpen", ["--filter", "sharpen"]),
    ("augrot_canny", ["--filter", "canny"]),
    ("augrot_feature_extraction", ["--freeze-features", "--lr", "1e-3"]),
    ("augrot_finetune_last4", "init:augrot_baseline"),
]
POST_STEPS = ("evaluate", "xai", "green")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOGS / "pipeline.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def keep_awake():
    #Windows: bez usypiania komputera w trakcie
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def server_up(uri) -> bool:
    try:
        return urllib.request.urlopen(f"{uri}/health", timeout=3).read() == b"OK"
    except OSError:
        return False


def ensure_server(uri):
    if server_up(uri):
        return
    log("MLflow server nie działa, uruchamiam")
    out = open(LOGS / "mlflow_server.log", "a", encoding="utf-8")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if os.name == "nt" else 0
    subprocess.Popen([sys.executable, "-m", "mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db",
                      "--default-artifact-root", f"file:{(ROOT / 'mlruns').as_posix()}", "--no-serve-artifacts",
                      "--host", "127.0.0.1", "--port", "5000"], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                     creationflags=flags)
    for _ in range(60):
        if server_up(uri):
            return
        time.sleep(2)
    raise SystemExit("MLflow server nie wstał")


def run(cmd, log_name) -> bool:
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "MLFLOW_DISABLE_AGENT_HINT": "1"}
    t = time.time()
    with open(LOGS / f"{log_name}.log", "w", encoding="utf-8") as f:
        code = subprocess.run([sys.executable, *cmd], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    log(f"{log_name}: kod {code}, {(time.time() - t) / 60:.1f} min")
    return code == 0


def find_run(name):
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT], filter_string=f"attributes.run_name = '{name}'",
                              order_by=["attributes.start_time DESC"])
    finished = runs[runs["status"] == "FINISHED"] if len(runs) else runs
    for rid in (runs[runs["status"] != "FINISHED"]["run_id"] if len(runs) else []):
        mlflow.MlflowClient().delete_run(rid)
        log(f"{name}: usunięty niedokończony run {rid}")
    return finished.iloc[0] if len(finished) else None


def train(name, extra, args):
    r = find_run(name)
    if r is None:
        if isinstance(extra, str):
            parent = find_run(extra.split(":", 1)[1])
            if parent is None:
                log(f"{name}: brak runu bazowego, pomijam")
                return None
            extra = ["--init-checkpoint", str(CKPT / parent["run_id"] / "best.pt"),
                     "--freeze-features", "--unfreeze-last", "4", "--lr", "1e-4"]
        log(f"{name}: trening")
        ok = run(["train.py", "--data-dir", str(DATA), "--experiment", EXPERIMENT, "--run-name", name,
                  "--tracking-uri", args.tracking_uri, "--num-workers", str(args.num_workers), *extra], f"{name}_train")
        r = find_run(name)
        if not ok or r is None:
            return None
    ckpt = CKPT / r["run_id"] / "best.pt"
    client = mlflow.MlflowClient()
    for step in POST_STEPS:
        if r.get(f"tags.pipeline_{step}") == "done":
            continue
        if run([f"{step}.py", "--checkpoint", str(ckpt), "--tracking-uri", args.tracking_uri], f"{name}_{step}"):
            client.set_tag(r["run_id"], f"pipeline_{step}", "done")
    return r


def old_dataset_tests(args):
    for label, (data, experiment) in OLD_TESTS.items():
        log(f"test na starym zbiorze ({label})")
        run(["eval_old_dataset.py", "--data-dir", str(ROOT / "dataset" / data), "--experiment", experiment,
             "--source-experiments", EXPERIMENT, "--tracking-uri", args.tracking_uri,
             "--out-dir", str(OUT / f"old_dataset_{label}")], f"old_dataset_{label}")


def comparison():
    def runs(experiments):
        df = mlflow.search_runs(experiment_names=experiments, filter_string="attributes.status = 'FINISHED'",
                                order_by=["attributes.start_time ASC"])
        return df

    train_runs = runs(["papier-kamien-baseline", "papier-kamien-bez-preprocessingu", "papier-kamien-preprocessing",
                       EXPERIMENT])
    exp_names = {e.experiment_id: e.name for e in mlflow.search_experiments()}
    rows = []
    for _, r in train_runs.iterrows():
        rows.append({
            "run": r["tags.mlflow.runName"], "run_id": r["run_id"], "eksperyment": exp_names[r["experiment_id"]],
            "augmentacja": r.get("params.augmentation"), "tryb_treningu": r.get("params.training_mode"),
            "filtr": r.get("params.image_filter"), "histeq": r.get("params.histogram_equalization"),
            "normalizacja": r.get("params.normalize"),
            "hagrid_test_accuracy": r.get("metrics.test_accuracy"), "hagrid_test_f1_macro": r.get("metrics.test_f1_macro"),
            "best_epoch": r.get("metrics.best_epoch"), "epoki": r.get("metrics.epochs_trained"),
            "train_min": r.get("metrics.train_duration_s", float("nan")) / 60,
            "train_energy_wh": r.get("metrics.train_energy_kwh", float("nan")) * 1000,
            "train_co2_g": r.get("metrics.train_co2_kg", float("nan")) * 1000,
            "infer_gpu_latency_ms": r.get("metrics.green_latency_gpu_b1_mean_ms"),
            "infer_cpu_latency_ms": r.get("metrics.green_latency_cpu_b1_mean_ms"),
            "infer_energy_wh_per_1000": r.get("metrics.green_infer_energy_total_wh_per_1000"),
            "infer_co2_g_per_1000": r.get("metrics.green_infer_co2_g_per_1000"),
            "trening_rownolegly": exp_names[r["experiment_id"]] == "papier-kamien-preprocessing",
        })
    df = pd.DataFrame(rows)
    for label, experiments in {"poziomo": ["papier-kamien-old-dataset-test", OLD_TESTS["poziomo"][1]],
                               "pionowo": ["papier-kamien-old-dataset-rot90-test", OLD_TESTS["pionowo"][1]]}.items():
        ev = runs(experiments)
        if not len(ev):
            continue
        ev = ev.drop_duplicates("params.source_run_id", keep="last").set_index("params.source_run_id")
        df[f"old_{label}_accuracy"] = df["run_id"].map(ev["metrics.test_accuracy"])
        df[f"old_{label}_f1_macro"] = df["run_id"].map(ev["metrics.test_f1_macro"])
    num = df.select_dtypes("float").columns
    df[num] = df[num].round(4)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "comparison_all.csv", index=False, encoding="utf-8-sig")

    cols = ["run", "hagrid_test_accuracy", "hagrid_test_f1_macro", "old_poziomo_f1_macro", "old_pionowo_f1_macro",
            "train_min", "train_energy_wh", "train_co2_g", "infer_energy_wh_per_1000"]
    md = df.sort_values("old_pionowo_f1_macro", ascending=False).reindex(columns=cols)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in md.itertuples(index=False)]
    (OUT / "comparison_all.md").write_text(
        "# Wszystkie modele: HaGRID test i stary zbiór (poziomo / obrócony pionowo)\n\n"
        "Modele z eksperymentu papier-kamien-preprocessing trenowane po 3 naraz, ich energia i CO2 są zawyżone.\n"
        "Modele augrot_* trenowane pojedynczo.\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    log(f"podsumowanie: {OUT / 'comparison_all.csv'}, {OUT / 'comparison_all.md'}")


def main():
    args = parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    keep_awake()
    ensure_server(args.tracking_uri)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    log("start")

    if not (DATA / "dataset_info.json").exists():
        log("buduję zbiór z pełnymi obrotami i odbiciami")
        if not run(["augment_dataset.py", "--preset", "rot", "--name", DATA.name, "--overwrite"], "augment_dataset"):
            raise SystemExit("augmentacja nie wyszła")

    failed = []
    for i, (name, extra) in enumerate(CONFIGS, 1):
        ensure_server(args.tracking_uri)
        log(f"[{i}/{len(CONFIGS)}] {name}")
        r = train(name, extra, args)
        if r is None:
            failed.append(name)
        else:
            log(f"{name}: HaGRID test acc {r['metrics.test_accuracy']:.4f}, f1 {r['metrics.test_f1_macro']:.4f}")

    ensure_server(args.tracking_uri)
    old_dataset_tests(args)
    comparison()
    log(f"koniec, nieudane: {failed or 'brak'}")


if __name__ == "__main__":
    main()
