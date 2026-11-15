# Test na starym zbiorze: przed obrotami i po obrotach

## Co było testowane

Modele trenowane na HaGRID (user split) sprawdziliśmy na starym zbiorze `dataset_origin`
(train + val + test razem, 2188 zdjęć: paper 712, rock 726, scissors 750).
Stary zbiór był testowany w dwóch wersjach:

- **poziomo**: zdjęcia w oryginalnej orientacji (`dataset/dataset_origin`), dłoń ułożona poziomo,
- **pionowo**: te same zdjęcia obrócone o 90° (`dataset/dataset_origin_rot90`), dłoń ułożona pionowo, tak jak w HaGRID.

## Pliki

- `przed_obrotami.csv`: 16 modeli trenowanych przed dodaniem obrotów
  (eksperymenty `papier-kamien-baseline`, `papier-kamien-bez-preprocessingu`, `papier-kamien-preprocessing`).
  Augmentacja offline w tych modelach to tylko obrót ±15°, odbicie poziome, zmiana kolorów, blur i grayscale.
  Modele `prep_*` i `baseline_*` nie miały augmentacji wcale.
- `po_obrotach.csv`: 13 modeli z eksperymentu `papier-kamien-aug-rotacje`. Zbiór treningowy był rozszerzony
  augmentacją offline x2 z obrotem o dowolny kąt (-180° do 180°, rogi wypełnione odbiciem obrazu),
  odbiciem poziomym i pionowym oraz tymi samymi zmianami koloru co wcześniej.

Wiersze są posortowane po średniej F1 z obu orientacji (`srednia_f1_poziomo_pionowo`), najlepsze na górze.

## Kolumny

| kolumna | znaczenie |
|---|---|
| model | nazwa runu w MLflow |
| augmentacja | `none`, `offline_aug_x2` (bez dużych obrotów), `offline_aug_rot_x2` (z obrotami) |
| tryb_treningu | `full_finetune`, `fine_tuning` (ostatnie 4 bloki), `feature_extraction` (tylko klasyfikator) |
| przetwarzanie | filtr obrazu przed siecią: none, histeq, clahe, gamma, grayscale, roi_center, blur, white_balance, sharpen, canny |
| normalizacja_imagenet | czy obraz był normalizowany średnią i odchyleniem ImageNet |
| hagrid_test_accuracy | accuracy na teście HaGRID, dla porównania |
| poziomo_* / pionowo_* | wyniki na starym zbiorze w danej orientacji |
| *_accuracy, *_f1_macro | accuracy i F1 makro |
| *_errors | liczba błędnie sklasyfikowanych zdjęć (na 2188) |
| *_f1_paper/rock/scissors | F1 dla każdej klasy |
| *_recall_paper/rock/scissors | recall dla każdej klasy (jaka część zdjęć danej klasy została rozpoznana) |
| *_mean_confidence | średnie prawdopodobieństwo przewidzianej klasy |
| srednia_f1_poziomo_pionowo | średnia F1 makro z obu orientacji |

## Najważniejsze wyniki

| | poziomo F1 (średnia / najlepszy) | pionowo F1 (średnia / najlepszy) |
|---|---|---|
| przed obrotami | 0.273 / 0.505 | 0.826 / 0.944 |
| po obrotach | 0.685 / 0.889 | 0.819 / 0.972 |

- Przed obrotami modele praktycznie nie działały na poziomych zdjęciach (F1 0.16 do 0.51, część na poziomie
  zgadywania). Na pionowych radziły sobie dobrze, bo w HaGRID dłonie są prawie zawsze pionowo.
  Problemem była więc orientacja dłoni, a nie sam stary zbiór.
- Po dodaniu obrotów wynik na poziomych zdjęciach wzrósł średnio z 0.27 do 0.69 F1. Na pionowych średnia
  została na podobnym poziomie (0.83 i 0.82), a najlepszy model jest lepszy niż wcześniej.
- Wynik na teście HaGRID prawie się nie zmienił (około 0.99), czyli obroty nie zepsuły modelu na danych, na których był uczony.
- Najlepszy model po obrotach to `augrot_grayscale`: F1 0.877 poziomo i 0.912 pionowo, najbardziej równy w obu orientacjach.
  `augrot_histeq` ma najlepszy wynik pionowo (0.972), ale poziomo tylko 0.685.
- Najsłabiej wypadają `augrot_feature_extraction` (zamrożona sieć, uczony tylko klasyfikator),
  `augrot_clahe` i `augrot_canny`.

## Para modeli przed i po (F1 makro)

Każdy model „po” zestawiony z najbliższym odpowiednikiem „przed” (ten sam filtr i tryb treningu).
W pierwszych czterech parach jedyna różnica to obroty w augmentacji. Modele `prep_*` nie miały augmentacji,
więc w ich parach różnica to augmentacja razem z obrotami.

| przed | po | poziomo przed | poziomo po | pionowo przed | pionowo po |
|---|---|---|---|---|---|
| mobilenet_v3_small_aug_offline | augrot_baseline | 0.332 | 0.749 | 0.903 | 0.798 |
| aug_offline_bez_preprocessingu | augrot_bez_normalizacji | 0.192 | 0.630 | 0.874 | 0.939 |
| aug_offline_histeq | augrot_histeq | 0.301 | 0.685 | 0.944 | 0.972 |
| finetune_aug_offline_last4 | augrot_finetune_last4 | 0.289 | 0.781 | 0.847 | 0.885 |
| prep_gamma | augrot_gamma | 0.237 | 0.889 | 0.877 | 0.794 |
| prep_clahe | augrot_clahe | 0.228 | 0.557 | 0.689 | 0.673 |
| prep_grayscale | augrot_grayscale | 0.254 | 0.876 | 0.883 | 0.912 |
| prep_roi_center | augrot_roi_center | 0.201 | 0.635 | 0.844 | 0.928 |
| prep_blur | augrot_blur | 0.252 | 0.705 | 0.881 | 0.813 |
| prep_white_balance | augrot_white_balance | 0.268 | 0.408 | 0.738 | 0.902 |
| prep_sharpen | augrot_sharpen | 0.382 | 0.844 | 0.747 | 0.679 |
| prep_canny | augrot_canny | 0.201 | 0.567 | 0.803 | 0.736 |

W każdej parze poziomo jest lepiej po obrotach. Pionowo w 6 parach jest lepiej, w 6 gorzej
(np. baseline 0.903 do 0.798, sharpen 0.747 do 0.679), więc model uczony na wszystkich kątach
częściowo traci przewagę na pionowych dłoniach w zamian za dużo lepsze działanie na poziomych.

## Skąd są dane

- przed: `outputs/old_dataset_test/metrics_old_dataset.csv` i `outputs/old_dataset_rot90_test/metrics_old_dataset.csv`
- po: `outputs/augrot/old_dataset_poziomo/metrics_old_dataset.csv` i `outputs/augrot/old_dataset_pionowo/metrics_old_dataset.csv`
- skrypt testu: `eval_old_dataset.py`, każdy test jest też osobnym runem w MLflow
  (`papier-kamien-old-dataset-test`, `papier-kamien-old-dataset-rot90-test`,
  `papier-kamien-augrot-old-dataset-test`, `papier-kamien-augrot-old-dataset-rot90-test`)
