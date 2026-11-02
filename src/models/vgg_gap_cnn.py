"""VGG-GAP CNN trenowana od zera (bez wag pretrenowanych).

wejście 3x224x224
-> 4 bloki VGG: [conv3x3 -> BN -> ReLU] x2 -> maxpool 2x2, kanały 32 -> 64 -> 128 -> 256  (wyjście 256x14x14)
-> global average pooling (256)
-> dropout 0.3 -> linear 256 -> num_classes
~1.17 M parametrów, podobnie jak MobileNetV3-Small (~1.52 M), żeby porównanie nie zależało od pojemności.
GAP zamiast flatten + dense (to byłoby >10 M parametrów przy ~1500 obrazach).
"""
from torch import nn


def vgg_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class VGGGapCNN(nn.Module):
    def __init__(self, num_classes: int, channels=(32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        blocks = []
        in_ch = 3
        for out_ch in channels:
            blocks.append(vgg_block(in_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_vgg_gap_cnn(num_classes: int, pretrained: bool) -> nn.Module:
    return VGGGapCNN(num_classes)
