#!/usr/bin/env bash
# MobileNetV3 A (feature extraction) -> B (fine-tuning z A) dla kilku seedów, potem dodatkowe seedy CNN.
# Uruchamiać z katalogu projektu: bash scripts/train_all.sh   (logi: outputs/logs/)
set -euo pipefail
export MLFLOW_DISABLE_AGENT_HINT=1
DATA=dataset/dataset_origin_new
run_id() {
  uv run python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///mlflow.db'); r = mlflow.search_runs(experiment_names=['papier-kamien'], filter_string=\"tags.mlflow.runName = '$1' and attributes.status = 'FINISHED'\", order_by=['start_time DESC']); print(r.run_id.iloc[0])"
}
for s in 42 1 2; do
  uv run python train.py --model mobilenet_v3_small --data-dir $DATA --freeze-features --lr 1e-3 \
    --seed $s --num-workers 4 --run-name mnv3_fe_s$s > outputs/logs/mnv3_fe_s$s.log 2>&1
  a=$(run_id mnv3_fe_s$s)
  uv run python train.py --model mobilenet_v3_small --data-dir $DATA --freeze-features --unfreeze-last 4 --bn-eval \
    --lr 1e-4 --seed $s --num-workers 4 --init-checkpoint outputs/checkpoints/$a/best.pt \
    --run-name mnv3_ft_s$s > outputs/logs/mnv3_ft_s$s.log 2>&1
  echo "seed $s: mnv3 A+B gotowe"
done
for s in 1 2; do
  uv run python train.py --data-dir $DATA --seed $s --num-workers 4 \
    --run-name baseline_vgg_gap_cnn_hagrid_s$s > outputs/logs/cnn_s$s.log 2>&1
  echo "seed $s: CNN gotowe"
done
