# Cały eksperyment po kolei (PowerShell, z katalogu projektu):  .\scripts\run_experiments.ps1
# Krok 1 pobiera dane, 2-3 trenują (GPU, długo). Krok 6 (external_test) uruchom ręcznie dopiero po wyborze modeli.
$ErrorActionPreference = "Stop"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
$Data = "dataset/dataset_origin_new"
$Ext = "dataset/external_hiw"
$Seeds = 42, 1, 2
$CnnRun = "82ff34fc1bcc4317a52068670b28939b"   # istniejący CNN (HaGRID), nie trenowany ponownie

function Get-RunId([string]$Name) {
    uv run python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///mlflow.db'); r = mlflow.search_runs(experiment_names=['papier-kamien'], filter_string=""tags.mlflow.runName = '$Name' and attributes.status = 'FINISHED'"", order_by=['start_time DESC']); print(r.run_id.iloc[0])"
}

# 1. Zewnętrzny zbiór (ok. 650 MB). Po zbudowaniu obejrzyj $Ext/preview_<klasa>.jpg.
if (-not (Test-Path $Ext)) { uv run python load_dataset.py hand-in-wild }
uv run pytest -q

# 2. MobileNetV3: A (feature extraction) i B (fine-tuning z A), kilka seedów
foreach ($s in $Seeds) {
    uv run python train.py --model mobilenet_v3_small --data-dir $Data --freeze-features --lr 1e-3 --seed $s --num-workers 4 --run-name "mnv3_fe_s$s"
    $a = Get-RunId "mnv3_fe_s$s"
    uv run python train.py --model mobilenet_v3_small --data-dir $Data --freeze-features --unfreeze-last 4 --bn-eval --lr 1e-4 --seed $s --num-workers 4 --init-checkpoint "outputs/checkpoints/$a/best.pt" --run-name "mnv3_ft_s$s"
}

# 3. (opcjonalnie) dodatkowe seedy CNN do średniej/std walidacji; istniejący checkpoint zostaje bez zmian
foreach ($s in 1, 2) {
    uv run python train.py --data-dir $Data --seed $s --num-workers 4 --run-name "baseline_vgg_gap_cnn_hagrid_s$s"
}

# 4. external_dev: istniejący CNN + MobileNet A/B (seed 42)
$Ckpts = @("outputs/checkpoints/$CnnRun/best.pt",
           "outputs/checkpoints/$(Get-RunId 'mnv3_fe_s42')/best.pt",
           "outputs/checkpoints/$(Get-RunId 'mnv3_ft_s42')/best.pt")
foreach ($c in $Ckpts) { uv run python evaluate.py --checkpoint $c --data-dir $Ext --split external_dev }

# 5. KOŃCOWY TEST: odkomentuj dopiero po zakończeniu analizy na external_dev i wyborze modeli:
# foreach ($c in $Ckpts) { uv run python evaluate.py --checkpoint $c --data-dir $Ext --split external_test --final-test }
