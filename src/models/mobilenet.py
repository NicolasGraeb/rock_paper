from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

_WEIGHTS_TF = MobileNet_V3_Small_Weights.DEFAULT.transforms()
MOBILENET_V3_SMALL_MEAN = tuple(_WEIGHTS_TF.mean)
MOBILENET_V3_SMALL_STD = tuple(_WEIGHTS_TF.std)


def build_mobilenet_v3_small(num_classes: int, pretrained: bool) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
