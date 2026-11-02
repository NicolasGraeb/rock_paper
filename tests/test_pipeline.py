"""Testy spójności: etykiety, splity, preprocessing, transfer learning, metryki.

    uv run pytest -q

Testy na danych pomijane, gdy danego datasetu/checkpointu nie ma na dysku.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from src.data import SPLITS, create_dataloaders, create_eval_loader
from src.engine import compute_classification_metrics, evaluate_model, set_train_mode
from src.models import MODELS, build_model, get_spec, set_trainable
from src.transforms import IMAGENET_MEAN, IMAGENET_STD, build_transforms, default_preprocessing

ROOT = Path(__file__).resolve().parents[1]
HAGRID = ROOT / "dataset" / "dataset_origin_new"
CLASSES = ["paper", "rock", "scissors"]
EXTERNAL = [d for d in (ROOT / "dataset").glob("external_*") if (d / "split.csv").exists()]

needs_hagrid = pytest.mark.skipif(not HAGRID.is_dir(), reason="brak dataset/dataset_origin_new")


# etykiety i splity

@needs_hagrid
def test_hagrid_classes_identical_in_all_splits():
    for s in SPLITS:
        assert sorted(p.name for p in (HAGRID / s).iterdir() if p.is_dir()) == CLASSES


@needs_hagrid
def test_hagrid_label_mapping_matches_source_gestures():
    info = json.loads((HAGRID / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["label_map"] == {"rock": "fist", "paper": "palm", "scissors": "peace"}
    split = pd.read_csv(HAGRID / "split.csv")
    expected = {"fist": "rock", "palm": "paper", "peace": "scissors"}
    assert (split["gesture"].map(expected) == split["label"]).all()
    # etykieta w ścieżce = etykieta w split.csv
    assert (split["path"].str.split("/").str[1] == split["label"]).all()


@needs_hagrid
def test_hagrid_split_csv_matches_disk_and_groups_not_shared():
    split = pd.read_csv(HAGRID / "split.csv")
    on_disk = {f.relative_to(HAGRID).as_posix() for s in SPLITS for f in (HAGRID / s).rglob("*") if f.is_file()}
    assert on_disk == set(split["path"])
    assert split["path"].is_unique
    assert split.groupby("group")["split"].nunique().max() == 1
    assert split.groupby("source")["split"].nunique().max() == 1  # ten sam obraz źródłowy tylko w jednym splicie


@needs_hagrid
def test_hagrid_loaders_use_augmentation_only_for_train():
    loaders, classes = create_dataloaders(HAGRID, batch_size=4)
    assert classes == CLASSES
    random_ops = (transforms.RandomHorizontalFlip, transforms.RandomRotation, transforms.ColorJitter)
    for s, loader in loaders.items():
        ops = loader.dataset.transform.transforms
        has_aug = any(isinstance(t, random_ops) for t in ops)
        assert has_aug == (s == "train"), s
        assert loader.dataset.class_to_idx == {c: i for i, c in enumerate(CLASSES)}


@pytest.mark.parametrize("ext", EXTERNAL, ids=lambda d: d.name)
def test_external_dataset_consistent(ext):
    split = pd.read_csv(ext / "split.csv")
    assert set(split["split"]) <= {"external_dev", "external_test"}
    assert set(split["label"]) <= set(CLASSES)
    assert split.groupby("group")["split"].nunique().max() == 1
    assert split["source_md5"].is_unique
    if HAGRID.is_dir():
        hagrid_md5 = {hashlib.md5(f.read_bytes()).hexdigest() for f in HAGRID.rglob("*.jp*g")}
        assert not (set(split["source_md5"]) & hagrid_md5), "plik zewnętrzny występuje w danych treningowych"


# preprocessing

def test_eval_transform_is_deterministic_and_normalized():
    from PIL import Image
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 200, 3), dtype=np.uint8))
    tf = build_transforms(224, train=False)
    a, b = tf(img), tf(img)
    assert a.shape == (3, 224, 224) and torch.equal(a, b)
    raw = transforms.ToTensor()(img.resize((224, 224), Image.BILINEAR))
    expected = (raw - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(IMAGENET_STD)[:, None, None]
    assert torch.allclose(a, expected, atol=0.05)


def test_mobilenet_normalization_matches_pretrained_weights():
    from torchvision.models import MobileNet_V3_Small_Weights
    ref = MobileNet_V3_Small_Weights.DEFAULT.transforms()
    spec = get_spec("mobilenet_v3_small")
    assert list(spec.mean) == list(ref.mean) and list(spec.std) == list(ref.std)


def test_old_checkpoints_default_to_imagenet_preprocessing():
    assert default_preprocessing(224) == {"image_size": 224, "resize": "square",
                                          "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)}


# transfer learning

def test_feature_extraction_trains_only_classifier():
    m = build_model("mobilenet_v3_small", 3, pretrained=False)
    set_trainable(m, freeze_features=True)
    trainable = {n.split(".")[0] for n, p in m.named_parameters() if p.requires_grad}
    assert trainable == {"classifier"}
    assert m.classifier[-1].out_features == 3


def test_fine_tuning_unfreezes_only_last_blocks():
    m = build_model("mobilenet_v3_small", 3, pretrained=False)
    set_trainable(m, freeze_features=True, unfreeze_last=3)
    n_blocks = len(m.features)
    for i, block in enumerate(m.features):
        assert all(p.requires_grad == (i >= n_blocks - 3) for p in block.parameters()), i


def test_frozen_batchnorm_stays_in_eval_and_keeps_stats():
    m = build_model("mobilenet_v3_small", 3, pretrained=False)
    set_trainable(m, freeze_features=True, unfreeze_last=2)
    set_train_mode(m, True)
    frozen_bn = [b for b in m.features[:-2].modules() if isinstance(b, nn.BatchNorm2d)]
    trainable_bn = [b for b in m.features[-2:].modules() if isinstance(b, nn.BatchNorm2d)]
    assert frozen_bn and all(not b.training for b in frozen_bn)
    assert trainable_bn and all(b.training for b in trainable_bn)
    before = frozen_bn[0].running_mean.clone()
    m(torch.randn(4, 3, 64, 64))
    assert torch.equal(before, frozen_bn[0].running_mean)
    set_train_mode(m, True, bn_eval=True)
    assert all(not b.training for b in m.modules() if isinstance(b, nn.BatchNorm2d))


# metryki

def test_loss_is_sample_weighted_over_whole_dataset():
    torch.manual_seed(0)
    x, y = torch.randn(10, 5), torch.randint(0, 3, (10,))
    model = nn.Linear(5, 3)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)  # batche 4, 4, 2
    metrics, y_true, y_pred = evaluate_model(model, loader, torch.device("cpu"), class_names=CLASSES)
    with torch.no_grad():
        full = nn.functional.cross_entropy(model(x), y).item()
        acc = (model(x).argmax(1) == y).float().mean().item()
    assert metrics["loss"] == pytest.approx(full, rel=1e-6)
    assert metrics["accuracy"] == pytest.approx(acc)


def test_confusion_matrix_keeps_all_classes_when_one_missing():
    m = compute_classification_metrics(np.array([0, 0, 1]), np.array([0, 2, 1]), class_names=CLASSES)
    assert np.array(m["confusion_matrix"]).shape == (3, 3)
    assert set(CLASSES) <= set(m["classification_report"])


def test_eval_loader_maps_labels_by_name(tmp_path):
    from PIL import Image
    for cls in ("rock", "scissors"):  # podzbiór klas, inna kolejność indeksów niż w ImageFolder
        (tmp_path / "external_dev" / cls).mkdir(parents=True)
        Image.new("RGB", (32, 32)).save(tmp_path / "external_dev" / cls / "a.jpg")
    loader = create_eval_loader(tmp_path, "external_dev", CLASSES, default_preprocessing(32))
    assert sorted(t for _, t in loader.dataset.samples) == [CLASSES.index("rock"), CLASSES.index("scissors")]
    with pytest.raises(ValueError):
        create_eval_loader(tmp_path, "external_dev", ["paper", "rock"], default_preprocessing(32))


def test_registry_models_build_with_three_classes():
    for name in MODELS:
        m = build_model(name, 3, pretrained=False)
        assert m(torch.randn(2, 3, 64, 64)).shape == (2, 3)
