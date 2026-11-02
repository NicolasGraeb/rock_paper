# Papier, kamień, nożyce: klasyfikacja gestów dłoni

## Setup

Wymagania: Python 3.12, [uv](https://docs.astral.sh/uv/), karta NVIDIA ze sterownikiem pod CUDA 12.8 (trening działa tylko na GPU),
[7-Zip](https://www.7-zip.org) (do rozpakowania zbioru zewnętrznego), ok. 6 GB miejsca na dysku.

    git clone https://github.com/NicolasGraeb/rock_paper.git
    cd rock_paper
    uv sync                                       # środowisko + torch z CUDA 12.8

    uv run python load_dataset.py hagrid          # pobiera HaGRID (~3.8 GB do data_cache/hagrid) i buduje dataset/dataset_origin_new
    uv run python load_dataset.py hand-in-wild    # pobiera Hand_in_Wild (~650 MB do data_cache/hand_in_wild) i buduje dataset/external_hiw

    uv run pytest -q                              # sprawdza etykiety, splity i preprocessing na zbudowanych danych

Po `hand-in-wild` obejrzyj `dataset/external_hiw/preview_<klasa>.jpg`, czy gesty pasują do etykiet.
Pobrane pliki zostają w `data_cache/`, więc ponowne uruchomienie (np. z `--overwrite`) nie ściąga ich drugi raz.
Seed podziału jest stały (`--seed 42`), więc na każdym komputerze powstaje identyczny podział.

Stary zbiór Kaggle (opcjonalnie): pobierz https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors,
rozpakuj do `archive/` (`archive/paper`, `archive/rock`, `archive/scissors`) i uruchom `uv run python load_dataset.py kaggle`.

## Dane

Wszystkie zbiory buduje jeden skrypt `load_dataset.py` (opcje: `uv run python load_dataset.py <źródło> --help`):

    uv run python load_dataset.py hagrid          # główny: HaGRID z HF (~3.8 GB) -> dataset/dataset_origin_new/{train,val,test}
    uv run python load_dataset.py hand-in-wild    # zewnętrzny test: Zenodo 5503489 (~650 MB) -> dataset/external_hiw/{external_dev,external_test}
    uv run python load_dataset.py folder --dir moje_zdjecia --name external_own   # własne zdjęcia: <klasa>/[<osoba>/]*.jpg
    uv run python load_dataset.py kaggle          # stary zbiór drgfreeman (rozpakowany w archive/) -> dataset/dataset_origin

- train/val/test: podział grupowy po seriach prawie identycznych zdjęć, stratyfikowany, ~70/15/15. HaGRID: fist/palm/peace
  -> rock/paper/scissors, klasy wyrównane do najmniejszej. Ograniczenie: te same osoby występują w train i test
  (78% zdjęć testu; accuracy na nowych osobach bez istotnej różnicy), patrz `notebooks/eda.ipynb`.
- zbiory zewnętrzne: tylko ewaluacja, podział external_dev/external_test po osobach/sesjach, kontrola nakładania
  z danymi treningowymi, `preview_<klasa>.jpg` do sprawdzenia gestów. W Hand_in_Wild „scissors” to w większości gest
  „L” (kciuk + wskazujący), `dataset/external_hiw/exclude_incompatible_scissors.csv` wyklucza niezgodne zdjęcia.

## Trening (tylko CUDA)

    uv run python train.py                                   # vgg_gap_cnn od zera, dataset/dataset_origin, early stopping (val_loss, patience 5)
    uv run python train.py --model mobilenet_v3_small        # pretrenowany MobileNetV3, cała sieć (lr domyślnie 3e-4)
    uv run python train.py --data-dir dataset/<wariant> --run-name <nazwa>

Transfer learning MobileNetV3-Small (ImageNet; normalizacja z wag modelu, zapisywana w checkpoincie):

    # A. feature extraction: zamrożone .features (BN w eval), trenowany .classifier
    uv run python train.py --model mobilenet_v3_small --data-dir dataset/dataset_origin_new --freeze-features --lr 1e-3 --run-name mnv3_fe
    # B. fine-tuning: start z best.pt wariantu A, odmrożone ostatnie 4 bloki .features, mniejszy lr, BN w eval
    uv run python train.py --model mobilenet_v3_small --data-dir dataset/dataset_origin_new --freeze-features --unfreeze-last 4 --bn-eval --lr 1e-4 --init-checkpoint outputs/checkpoints/<run_id_A>/best.pt --run-name mnv3_ft

Kilka seedów: dopisz `--seed N` i sufiks `_s<N>` do `--run-name` (porównanie w MLflow UI).
Całość po kolei: `scripts/run_experiments.ps1`.

### Ewaluacja

    uv run python evaluate.py --checkpoint outputs/checkpoints/<run_id>/best.pt --split test                   # HaGRID test
    uv run python evaluate.py --checkpoint ... --data-dir dataset/external_hiw --split external_dev           # diagnostyka
    uv run python evaluate.py --checkpoint ... --data-dir dataset/external_hiw --split external_dev --exclude-csv dataset/external_hiw/exclude_incompatible_scissors.csv
    # końcowy test: raz, po wyborze modeli; nie dobierać potem niczego na jego podstawie:
    uv run python evaluate.py --checkpoint ... --data-dir dataset/external_hiw --split external_test --final-test

Do MLflow (eksperymenty `papier-kamien-external-dev` / `papier-kamien-external-test` / `papier-kamien-eval`) trafiają:
metryki globalne i per klasa, macierz pomyłek, `predictions_<split>.csv` (ścieżka, klasa prawdziwa/przewidziana,
prawdopodobieństwa, poprawność, jasność/kontrast/rozmiar), `errors_<split>.png` (błędy z confidence) i `ms_per_image`
(forward pass, batch 32, 224 px, warm-up + synchronizacja CUDA).

### Testy

    uv run pytest -q

### Nowy model
1. Plik `src/models/<nazwa>.py` z funkcją `build_<nazwa>(num_classes, pretrained) -> nn.Module`.
2. Wpis w `MODELS` w `src/models/__init__.py` (czy ma wagi pretrenowane, domyślny lr, normalizacja).
3. `uv run python train.py --model <nazwa>`: wykresy, macierz pomyłek, checkpointy i early stopping działają bez zmian.

## Gdzie są wyniki

- `outputs/checkpoints/<run_id>/`: `best.pt`, `last.pt` i `learning_curves.png` (krzywa uczenia odświeżana co epokę)
- `mlruns/1/<run_id>/artifacts/`: dla każdego treningu krzywe uczenia, `confusion_matrix_{val,test}.png`,
  `classification_report_{val,test}.json`, `metrics.json`, `history.json`, `model.pt`
- `mlruns/4/<run_id>/artifacts/`: ewaluacja na external_dev: macierz pomyłek, `errors_external_dev.png`, `predictions_external_dev.csv`
- `outputs/comparison_*.md`: tabele porównawcze modeli, `outputs/logs/`: logi treningów
- najwygodniej przez MLflow UI (niżej), `http://127.0.0.1:5000`

## MLflow

    uv run mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
