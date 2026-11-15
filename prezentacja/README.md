# Eksperymenty: papier, kamień, nożyce

Wszystkie runy z MLflow, od pierwszego baseline (2.11.2026) do testów augmentacji z obrotami (11.11.2026, 2:07).
87 runów w 8 eksperymentach. Pominięte są eksperymenty i runy usunięte w MLflow (smoke testy, stare próby).

## Zawartość folderu

| plik / folder | co jest w środku |
|---|---|
| `runy_podsumowanie.csv` | jeden wiersz na run: eksperyment, run, run_id, czas, wszystkie parametry (`param.*`), wszystkie metryki końcowe |
| `metryki_per_epoka.csv` | jeden wiersz na epokę: train/val loss, accuracy, F1, precision, recall |
| `metryki_wszystkie_long.csv` | surowe metryki z MLflow (metryka, krok, wartość), w tym `train_loss_step` co 10 kroków |
| `artefakty/<eksperyment>/<run>/` | kopia artefaktów runu z MLflow |

## Wspólne ustawienia

Wszystkie modele to MobileNetV3-Small z wagami ImageNet (1.52 mln parametrów), 3 klasy (paper, rock, scissors).

- wejście 224x224, resize do kwadratu bez zachowania proporcji
- normalizacja ImageNet (mean 0.485/0.456/0.406, std 0.229/0.224/0.225), chyba że run mówi inaczej
- AdamW, lr 3e-4, weight decay 1e-4, batch 32, seed 42
- maks. 50 epok, early stopping na `val_loss` z patience 5, zapisywany najlepszy checkpoint (`best_epoch`)
- GPU: RTX 5070 Ti, energia i CO2 z codecarbon

### Zbiory danych

**hagrid_user_split** (zbiór główny). Wycinek HaGRID (`cj-mills/hagrid-classification-512p-no-gesture-150k`),
klasy fist → rock, palm → paper, peace → scissors. 6870 obrazów na klasę, 20 610 razem. Podział 70/15/15 po
użytkownikach (`user_id` z adnotacji HaGRID), więc ta sama osoba nie trafia do dwóch splitów.

| split | paper | rock | scissors | razem |
|---|---|---|---|---|
| train | 4908 | 4909 | 4908 | 14 725 |
| val | 982 | 981 | 982 | 2 945 |
| test | 980 | 980 | 980 | 2 940 |

**hagrid_user_split_aug** (augmentacja offline x2). Train = oryginały + 2 zaugmentowane kopie każdego zdjęcia
(44 175 obrazów). Val i test bez zmian. Augmentacja (`augment_dataset.py --preset small`):
- odbicie poziome p=0.5
- obrót ±15°, przesunięcie do 5%, skala 0.9-1.1
- color jitter: jasność, kontrast, nasycenie 0.3, odcień 0.05
- rozmycie Gaussa p=0.2 (kernel 5, sigma 0.1-1.5)
- skala szarości p=0.1

**hagrid_user_split_aug_rot** (augmentacja offline x2 z pełnymi obrotami). Jak wyżej, ale (`--preset rot`):
- obrót o dowolny kąt od -180° do 180°, rogi wypełnione odbiciem obrazu zamiast czarnego tła
- odbicie poziome p=0.5 i pionowe p=0.5
- przesunięcie do 5%, skala 0.9-1.1, ten sam color jitter, blur i grayscale

Train ma 44 175 obrazów, val i test bez zmian.

**dataset_origin** (stary zbiór, test zewnętrzny). Zbiór rps-cv-images (Julien de la Bruère-Terreault,
CC-BY-SA 4.0): 2188 zdjęć 300x200 na zielonym tle, przy stałym oświetleniu. Dłoń wchodzi z boku kadru (poziomo).
Model go nigdy nie widział w treningu. Do testu brane są wszystkie zdjęcia (train+val+test starego podziału).

**dataset_origin_rot90**. Ten sam stary zbiór obrócony o 90° zgodnie z ruchem wskazówek zegara, więc dłoń
jest pionowo, palcami do góry, jak w HaGRID.

### Filtry przetwarzania obrazu

Filtr działa na każdym splicie (train, val, test) i przy teście na starym zbiorze. Wszystkie oprócz `roi_center`
działają na obrazie po resize do 224x224.

| filtr | co robi |
|---|---|
| `none` | nic |
| `histeq` | wyrównanie histogramu tylko na jasności (kanał Y z YCrCb), kolory zostają |
| `clahe` | CLAHE na jasności, clipLimit 2.0, siatka 8x8 |
| `gamma` | korekcja gamma dobierana per obraz, tak żeby średnia jasność wyszła ok. 0.5 (gamma 0.4-2.5) |
| `grayscale` | skala szarości skopiowana na 3 kanały |
| `roi_center` | wycięcie środkowego kwadratu 80% krótszego boku, przed resize |
| `blur` | rozmycie Gaussa, sigma 1.5 |
| `white_balance` | balans bieli gray world |
| `sharpen` | wyostrzenie unsharp mask (2·obraz − rozmyty, sigma 1.0) |
| `canny` | krawędzie Canny (progi 50/150) zamiast obrazu |

### Jak testowano

1. **HaGRID test** (`test_*` w runach treningowych): 2940 zdjęć z test splitu, osoby nieobecne w train i val.
   Liczone na najlepszym checkpoincie.
2. **Stary zbiór poziomo** (`papier-kamien-old-dataset-test`, `papier-kamien-augrot-old-dataset-test`):
   najlepszy checkpoint każdego modelu na całym `dataset_origin` (2188 zdjęć), z tym samym preprocessingiem co
   w treningu. `test_accuracy` dotyczy całości, `test_accuracy_old_train/val/test` to wynik na częściach starego podziału.
3. **Stary zbiór pionowo** (`...-rot90-test`): to samo na `dataset_origin_rot90`.

Runy testowe na starym zbiorze nie trenują niczego. Mają tę samą nazwę co run źródłowy, a jego id jest
w `param.source_run_id`. Metryka `hagrid_test_accuracy` to dla porównania wynik źródłowego modelu na HaGRID.

## Eksperymenty

### 1. papier-kamien-baseline (2-3.11)

Pierwszy baseline i pierwsze warianty, wszystko na HaGRID.

| run | zbiór treningowy | preprocessing | trening | uwagi |
|---|---|---|---|---|
| `baseline_mobilenet_v3_small` | hagrid_user_split | normalizacja ImageNet | cały model | pierwszy baseline; jedyny run z ewaluacją, XAI, green i eksportem ONNX |
| `mobilenet_v3_small_aug_offline` | hagrid_user_split_aug | normalizacja ImageNet | cały model | augmentacja offline x2 |
| `baseline_histeq` | hagrid_user_split | histeq + normalizacja | cały model | |
| `aug_offline_histeq` | hagrid_user_split_aug | histeq + normalizacja | cały model | |
| `finetune_aug_offline_last4` | hagrid_user_split_aug | normalizacja ImageNet | start z `best.pt` runu `mobilenet_v3_small_aug_offline`, zamrożony ekstraktor bez ostatnich 4 bloków, lr 1e-4 | 1.33 mln trenowalnych parametrów |

### 2. papier-kamien-bez-preprocessingu (3.11)

Sprawdzenie, co daje normalizacja ImageNet. Piksele tylko przeskalowane do 0-1 (mean 0, std 1).

| run | zbiór treningowy | preprocessing |
|---|---|---|
| `baseline_bez_preprocessingu` | hagrid_user_split | brak normalizacji |
| `aug_offline_bez_preprocessingu` | hagrid_user_split_aug | brak normalizacji |

### 3. papier-kamien-preprocessing (9.11)

Porównanie filtrów obrazu (`run_preprocessing.py`). Zbiór hagrid_user_split bez augmentacji, normalizacja ImageNet,
cały model trenowany. Run `prep_<filtr>` używa filtra z tabeli wyżej, `prep_none` to punkt odniesienia.
Runy: `prep_none`, `prep_clahe`, `prep_gamma`, `prep_grayscale`, `prep_roi_center`, `prep_blur`,
`prep_white_balance`, `prep_sharpen`, `prep_canny`.

Treningi szły po 3 równolegle na jednym GPU, więc czas, energia i CO2 tych runów są zawyżone i nie da się
ich porównać z resztą.

### 4. papier-kamien-old-dataset-test (10.11, 23:31)

Test 16 modeli z eksperymentów 1-3 na starym zbiorze poziomo (`eval_old_dataset.py`). Bez treningu.

### 5. papier-kamien-old-dataset-rot90-test (10.11, 23:40)

To samo na starym zbiorze obróconym pionowo.

### 6. papier-kamien-aug-rotacje (11.11, 0:12-2:00)

Te same konfiguracje co wcześniej, ale wszystkie na zbiorze **hagrid_user_split_aug_rot** (pełne obroty i odbicia),
`run_augrot.py`. Treningi po kolei, jeden naraz, więc czas i energia są wiarygodne. Każdy run ma też ewaluację,
Grad-CAM, occlusion i pomiary green.

| run | preprocessing / trening |
|---|---|
| `augrot_baseline` | normalizacja ImageNet, cały model |
| `augrot_bez_normalizacji` | bez normalizacji |
| `augrot_histeq` | histeq |
| `augrot_clahe`, `augrot_gamma`, `augrot_grayscale`, `augrot_roi_center`, `augrot_blur`, `augrot_white_balance`, `augrot_sharpen`, `augrot_canny` | odpowiedni filtr z tabeli |
| `augrot_feature_extraction` | zamrożony cały ekstraktor, trenowany tylko klasyfikator, lr 1e-3 |
| `augrot_finetune_last4` | start z `best.pt` runu `augrot_baseline`, odmrożone ostatnie 4 bloki, lr 1e-4 |

### 7. papier-kamien-augrot-old-dataset-test (11.11, 2:04)

13 modeli z eksperymentu 6 na starym zbiorze poziomo.

### 8. papier-kamien-augrot-old-dataset-rot90-test (11.11, 2:06)

13 modeli z eksperymentu 6 na starym zbiorze pionowo.

## Wyniki zbiorczo (F1 macro)

HaGRID = test split HaGRID, stary poziomo / pionowo = cały dataset_origin / dataset_origin_rot90.

| eksperyment | run | HaGRID | stary poziomo | stary pionowo | epoki (najlepsza) |
|---|---|---|---|---|---|
| baseline | baseline_mobilenet_v3_small | 0.9946 | 0.1649 | 0.9070 | 9 (4) |
| baseline | mobilenet_v3_small_aug_offline | 0.9963 | 0.3316 | 0.9030 | 7 (2) |
| baseline | baseline_histeq | 0.9912 | 0.5054 | 0.7459 | 10 (5) |
| baseline | aug_offline_histeq | 0.9952 | 0.3007 | 0.9444 | 7 (2) |
| baseline | finetune_aug_offline_last4 | 0.9969 | 0.2895 | 0.8465 | 6 (1) |
| bez-preprocessingu | baseline_bez_preprocessingu | 0.9956 | 0.2920 | 0.7851 | 9 (4) |
| bez-preprocessingu | aug_offline_bez_preprocessingu | 0.9952 | 0.1919 | 0.8736 | 7 (2) |
| preprocessing | prep_none | 0.9956 | 0.2674 | 0.7460 | 11 (6) |
| preprocessing | prep_clahe | 0.9942 | 0.2281 | 0.6892 | 15 (10) |
| preprocessing | prep_gamma | 0.9946 | 0.2368 | 0.8774 | 8 (3) |
| preprocessing | prep_grayscale | 0.9918 | 0.2543 | 0.8835 | 14 (9) |
| preprocessing | prep_roi_center | 0.9946 | 0.2011 | 0.8440 | 8 (3) |
| preprocessing | prep_blur | 0.9949 | 0.2521 | 0.8808 | 8 (3) |
| preprocessing | prep_white_balance | 0.9939 | 0.2675 | 0.7379 | 15 (10) |
| preprocessing | prep_sharpen | 0.9959 | 0.3819 | 0.7467 | 18 (13) |
| preprocessing | prep_canny | 0.9500 | 0.2009 | 0.8032 | 9 (4) |
| aug-rotacje | augrot_baseline | 0.9935 | 0.7487 | 0.7984 | 7 (2) |
| aug-rotacje | augrot_bez_normalizacji | 0.9939 | 0.6298 | 0.9394 | 9 (4) |
| aug-rotacje | augrot_histeq | 0.9888 | 0.6854 | 0.9721 | 7 (2) |
| aug-rotacje | augrot_clahe | 0.9939 | 0.5572 | 0.6730 | 11 (6) |
| aug-rotacje | augrot_gamma | 0.9932 | 0.8887 | 0.7945 | 8 (3) |
| aug-rotacje | augrot_grayscale | 0.9861 | 0.8765 | 0.9120 | 8 (3) |
| aug-rotacje | augrot_roi_center | 0.9946 | 0.6347 | 0.9280 | 8 (3) |
| aug-rotacje | augrot_blur | 0.9939 | 0.7051 | 0.8134 | 11 (6) |
| aug-rotacje | augrot_white_balance | 0.9925 | 0.4084 | 0.9024 | 7 (2) |
| aug-rotacje | augrot_sharpen | 0.9932 | 0.8436 | 0.6791 | 8 (3) |
| aug-rotacje | augrot_canny | 0.9368 | 0.5668 | 0.7356 | 8 (3) |
| aug-rotacje | augrot_feature_extraction | 0.9223 | 0.5827 | 0.6158 | 9 (4) |
| aug-rotacje | augrot_finetune_last4 | 0.9949 | 0.7813 | 0.8853 | 6 (1) |

Co z tego wynika:
- Na HaGRID prawie wszystko daje 0.99+, więc ten test nie rozróżnia modeli. Odstają tylko `canny` i `feature_extraction`.
- Modele bez pełnych obrotów prawie nie działają na starym zbiorze poziomo (F1 0.16-0.51), a po obróceniu
  zdjęć do pionu dają 0.69-0.94. Głównym problemem jest więc orientacja dłoni, a nie samo tło czy oświetlenie.
- Augmentacja z pełnymi obrotami podnosi wynik poziomo do 0.41-0.89, a HaGRID spada zwykle o mniej niż 0.01
  (najwięcej `canny`, o 0.013).
  Najlepsze poziomo: `augrot_gamma` 0.889 i `augrot_grayscale` 0.877. Najlepsze pionowo: `augrot_histeq` 0.972.
  `augrot_grayscale` jest najrówniejszy w obu orientacjach (0.877 / 0.912).

## Artefakty w runie

Runy treningowe:

| plik | opis |
|---|---|
| `model.pt` | najlepszy checkpoint |
| `metrics.json`, `history.json` | metryki końcowe i historia per epoka |
| `learning_curves.png` | krzywe loss i accuracy |
| `confusion_matrix_val.png`, `confusion_matrix_test.png`, `classification_report_*.json` | wyniki na val i test HaGRID |
| `energy.json` | energia i CO2 treningu |
| `dataset/dataset_info.json`, `dataset/split.csv` | opis zbioru i lista plików z przypisaniem do splitu |
| `eval/` | predykcje na teście, lista błędów, `errors_test.png` z najpewniejszymi pomyłkami |
| `xai/` | Grad-CAM i occlusion dla poprawnych i błędnych przykładów |
| `green/` | latencja CPU/GPU, FPS, pamięć, energia na 1000 predykcji |
| `export/` | `model.onnx`, `preprocessing.json` i sprawdzenie zgodności ONNX z PyTorch (tylko pierwszy baseline) |

`eval/`, `xai/` i `green/` mają pierwszy baseline i wszystkie runy `augrot_*`. Pozostałe runy treningowe mają tylko
artefakty z samego treningu.

Runy testowe na starym zbiorze:

| plik | opis |
|---|---|
| `eval/predictions_old_dataset.csv` | predykcja dla każdego zdjęcia z pewnością i prawdopodobieństwami klas |
| `eval/confusion_matrix_old_dataset.png`, `classification_report_old_dataset.json`, `metrics_old_dataset.json` | wyniki |
| `eval/errors_old_dataset.png` | najpewniejsze pomyłki |
| `xai/gradcam_correct.png`, `xai/gradcam_errors.png` | Grad-CAM, 3 przykłady na klasę |
