# Przetwarzanie obrazu, eksperyment papier-kamien-preprocessing

MobileNetV3-Small, hagrid_user_split, bez augmentacji, seed 42. Filtr na wszystkich splitach.
Treningi szły równolegle, więc czas i energia z codecarbon są zawyżone.

| filtr | run | val acc | val F1 macro | test acc | test F1 macro | najlepsza epoka | czas treningu [min] |
|---|---|---|---|---|---|---|---|
| sharpen | 67af0d9a | 0.9929 | 0.9929 | 0.9959 | 0.9959 | 13 | 11.7 |
| none | b2193f21 | 0.9929 | 0.9929 | 0.9956 | 0.9956 | 6 | 10.4 |
| blur | f88444e0 | 0.9874 | 0.9874 | 0.9949 | 0.9949 | 3 | 8.7 |
| gamma | 4e3da81a | 0.9898 | 0.9898 | 0.9946 | 0.9946 | 3 | 6.0 |
| roi_center | d8e4aee8 | 0.9952 | 0.9952 | 0.9946 | 0.9946 | 3 | 7.8 |
| clahe | 11362e38 | 0.9925 | 0.9925 | 0.9942 | 0.9942 | 10 | 10.6 |
| white_balance | 1a7736d1 | 0.9929 | 0.9929 | 0.9939 | 0.9939 | 10 | 15.3 |
| grayscale | 5cd76329 | 0.9912 | 0.9912 | 0.9918 | 0.9918 | 9 | 8.8 |
| canny | 5fbedcac | 0.9328 | 0.9327 | 0.9500 | 0.9500 | 4 | 7.8 |
