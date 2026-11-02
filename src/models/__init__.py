"""Rejestr modeli. Nowy model = nowy plik w src/models/ + jeden wpis w MODELS.

Builder ma sygnaturę (num_classes, pretrained) -> nn.Module.
"""
from dataclasses import dataclass
from typing import Callable

from torch import nn

from src.models.mobilenet import MOBILENET_V3_SMALL_MEAN, MOBILENET_V3_SMALL_STD, build_mobilenet_v3_small
from src.models.vgg_gap_cnn import build_vgg_gap_cnn
from src.transforms import IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class ModelSpec:
    builder: Callable[[int, bool], nn.Module]
    supports_pretrained: bool  # czy istnieją wagi pretrenowane (ImageNet)
    default_lr: float          # używany, gdy nie podano --lr
    mean: tuple = IMAGENET_MEAN  # normalizacja wejścia; dla modeli pretrenowanych ta z ich wag
    std: tuple = IMAGENET_STD


MODELS = {
    "mobilenet_v3_small": ModelSpec(build_mobilenet_v3_small, supports_pretrained=True, default_lr=3e-4,
                                    mean=MOBILENET_V3_SMALL_MEAN, std=MOBILENET_V3_SMALL_STD),
    # trenowany od zera; ImageNet mean/std użyte historycznie, zostaje dla zgodności z checkpointami
    "vgg_gap_cnn": ModelSpec(build_vgg_gap_cnn, supports_pretrained=False, default_lr=1e-3),
}


def get_spec(name: str) -> ModelSpec:
    if name not in MODELS:
        raise ValueError(f"Nieznany model {name!r}. Dostępne: {', '.join(MODELS)}")
    return MODELS[name]


def build_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    spec = get_spec(name)
    if pretrained and not spec.supports_pretrained:
        raise ValueError(f"Model {name!r} nie ma wag pretrenowanych, użyj --no-pretrained")
    return spec.builder(num_classes, pretrained)


def set_trainable(model: nn.Module, freeze_features: bool, unfreeze_last: int = 0) -> None:
    """Transfer learning dla modeli z `.features` (ekstraktor cech) i `.classifier` (głowa).

    freeze_features=True, unfreeze_last=0 -> feature extraction: uczy się tylko `.classifier`.
    unfreeze_last=k -> fine-tuning: dodatkowo ostatnie k bloków `.features`.
    """
    if not freeze_features:
        return
    if not hasattr(model, "features"):
        raise ValueError(f"{type(model).__name__} nie ma atrybutu .features, nie da się zamrozić ekstraktora")
    for p in model.features.parameters():
        p.requires_grad = False
    if unfreeze_last:
        if unfreeze_last > len(model.features):
            raise ValueError(f"unfreeze_last={unfreeze_last} > liczba bloków features ({len(model.features)})")
        for block in model.features[-unfreeze_last:]:
            for p in block.parameters():
                p.requires_grad = True


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
