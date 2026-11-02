# Papier, kamień, nożyce

Klasyfikacja gestów dłoni (MobileNetV3-Small, zbiór HaGRID) i aplikacja Android z przednią kamerą.

## 1. Wymagania

- Windows albo Linux, karta NVIDIA ze sterownikiem obsługującym CUDA 12.8 (trening działa tylko na GPU)
- git
- ok. 12 GB wolnego miejsca (pobrane dane ~4.6 GB, zbiór ~0.5 GB, środowisko Pythona ~5.2 GB)
- internet przy pierwszym uruchomieniu

Python instalować nie trzeba, uv pobierze właściwą wersję (3.12) sam.

## 2. Instalacja

Zainstaluj uv:

    #Windows (PowerShell)
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    #Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

Otwórz nowy terminal, potem:

    git clone https://github.com/NicolasGraeb/rock_paper.git
    cd rock_paper
    uv sync

Sprawdź, czy PyTorch widzi kartę graficzną (ma wypisać `True` i nazwę karty):

    uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

## 3. Dane

    uv run python load_dataset.py

Pobiera HaGRID z Hugging Face (~3.8 GB) i adnotacje z `user_id` (~720 MB) do `data_cache/`, a potem buduje
`dataset/hagrid_user_split/{train,val,test}`. Trwa kilkanaście minut. Jeśli pobieranie się przerwie, uruchom
ponownie: pobrane pliki zostają w `data_cache/`. Podział jest stały (seed 42), więc wychodzi zawsze ten sam.

Sprawdzenie danych i kodu:

    uv run pytest -q

## 4. Trening

    uv run python train.py

Trenuje baseline (MobileNetV3-Small z wagami ImageNet, bez augmentacji) z early stoppingiem. Na RTX 5070 Ti trwa
ok. 7 minut. Na koniec wypisuje `run_id` oraz wynik na val i test. Checkpoint: `outputs/baseline/checkpoints/<run_id>/best.pt`.

Opcje: `uv run python train.py --help` (np. `--epochs`, `--batch-size`, `--lr`, `--seed`, `--run-name`).

## 5. Ewaluacja, XAI, Green AI, eksport

Bez `--checkpoint` każdy skrypt bierze najnowszy `best.pt`. Wyniki trafiają do `outputs/baseline/results/<run_id>/`
i do tego samego runu w MLflow.

    uv run python evaluate.py    #metryki, confusion matrix, błędy
    uv run python xai.py         #Grad-CAM i occlusion
    uv run python green.py       #czas, pamięć, energia, CO2
    uv run python export.py      #ONNX + kopia do apki

## 6. MLflow

    uv run mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

Potem w przeglądarce http://127.0.0.1:5000, eksperyment `papier-kamien-baseline`.

## 7. Aplikacja Android

Potrzebne: Android Studio (z Android SDK), JDK 17, telefon z Androidem 11 lub nowszym.

1. Uruchom `uv run python export.py`, żeby model trafił do aplikacji.
2. Na telefonie włącz opcje programisty i debugowanie USB, podłącz telefon kablem i zaakceptuj komunikat.
3. Zainstaluj aplikację:

       cd android
       ./gradlew installDebug        #Windows: gradlew.bat installDebug

   Albo otwórz katalog `android/` w Android Studio i kliknij Run.

Testy aplikacji (drugi wymaga podłączonego telefonu i wcześniejszego `export.py`):

    ./gradlew testDebugUnitTest
    ./gradlew connectedDebugAndroidTest

Aplikacja klasyfikuje środek kadru i nie wykrywa dłoni, więc zawsze pokazuje jedną z trzech klas.

## Problemy

- `CUDA niedostępna`: zaktualizuj sterownik NVIDIA i sprawdź polecenie z punktu 2.
- `dataset/hagrid_user_split już istnieje`: zbiór jest gotowy; żeby zbudować go od nowa, dodaj `--overwrite`.
- `Brak checkpointów`: najpierw uruchom `train.py`.
- Gradle nie widzi Javy: ustaw `JAVA_HOME` na katalog JDK 17.
