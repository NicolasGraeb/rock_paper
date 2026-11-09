# Wszystkie modele na teście

MobileNetV3-Small, hagrid_user_split, test 2940 obrazów, seed 42. Eksperymenty MLflow: papier-kamien-baseline, papier-kamien-bez-preprocessingu, papier-kamien-preprocessing.

| # | run | augmentacja | trening | przetwarzanie | test acc | test F1 macro | błędy | F1 paper | F1 rock | F1 scissors | test loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | finetune_aug_offline_last4 | offline x2 | fine-tuning ost. 4 bloków | ImageNet norm | 0.9969 | 0.9969 | 9 | 0.9969 | 0.9974 | 0.9964 | 0.0114 |
| 2 | mobilenet_v3_small_aug_offline | offline x2 | pełny | ImageNet norm | 0.9963 | 0.9963 | 11 | 0.9959 | 0.9969 | 0.9959 | 0.0118 |
| 3 | prep_sharpen | brak | pełny | unsharp mask | 0.9959 | 0.9959 | 12 | 0.9964 | 0.9969 | 0.9944 | 0.0141 |
| 4 | baseline_bez_preprocessingu | brak | pełny | bez normalizacji | 0.9956 | 0.9956 | 13 | 0.9954 | 0.9974 | 0.9939 | 0.0149 |
| 5 | prep_none | brak | pełny | ImageNet norm | 0.9956 | 0.9956 | 13 | 0.9954 | 0.9974 | 0.9939 | 0.0176 |
| 6 | aug_offline_bez_preprocessingu | offline x2 | pełny | bez normalizacji | 0.9952 | 0.9952 | 14 | 0.9959 | 0.9964 | 0.9934 | 0.0128 |
| 7 | aug_offline_histeq | offline x2 | pełny | wyrównanie histogramu | 0.9952 | 0.9952 | 14 | 0.9964 | 0.9949 | 0.9944 | 0.0166 |
| 8 | prep_blur | brak | pełny | gaussian blur | 0.9949 | 0.9949 | 15 | 0.9934 | 0.9974 | 0.9939 | 0.0163 |
| 9 | baseline_mobilenet_v3_small | brak | pełny | ImageNet norm | 0.9946 | 0.9946 | 16 | 0.9949 | 0.9954 | 0.9934 | 0.0157 |
| 10 | prep_gamma | brak | pełny | gamma | 0.9946 | 0.9946 | 16 | 0.9964 | 0.9944 | 0.9929 | 0.0162 |
| 11 | prep_roi_center | brak | pełny | crop środka 80% | 0.9946 | 0.9946 | 16 | 0.9954 | 0.9944 | 0.9939 | 0.0161 |
| 12 | prep_clahe | brak | pełny | CLAHE | 0.9942 | 0.9942 | 17 | 0.9954 | 0.9944 | 0.9928 | 0.0149 |
| 13 | prep_white_balance | brak | pełny | white balance (gray world) | 0.9939 | 0.9939 | 18 | 0.9944 | 0.9949 | 0.9923 | 0.0244 |
| 14 | prep_grayscale | brak | pełny | skala szarości | 0.9918 | 0.9918 | 24 | 0.9924 | 0.9944 | 0.9888 | 0.0314 |
| 15 | baseline_histeq | brak | pełny | wyrównanie histogramu | 0.9912 | 0.9912 | 26 | 0.9914 | 0.9934 | 0.9887 | 0.0281 |
| 16 | prep_canny | brak | pełny | krawędzie Canny | 0.9500 | 0.9500 | 147 | 0.9519 | 0.9540 | 0.9440 | 0.1418 |

Błędy = (1 - acc) * 2940. Jeden seed, więc różnice kilku obrazów mieszczą się w szumie.
