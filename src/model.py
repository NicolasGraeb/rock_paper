from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

MODEL_NAME = "mobilenet_v3_small"


def build_model(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
