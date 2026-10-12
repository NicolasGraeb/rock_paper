from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
):
    train = optimizer is not None
    model.train(train)
    losses = []
    preds_all = []
    targets_all = []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if train:
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        preds_all.append(logits.argmax(dim=1).detach().cpu().numpy())
        targets_all.append(targets.detach().cpu().numpy())

    y_pred = np.concatenate(preds_all)
    y_true = np.concatenate(targets_all)
    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean(losses))
    return metrics, y_true, y_pred


def compute_classification_metrics(y_true, y_pred, class_names=None):
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": report,
    }
    return metrics


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names=None,
):
    criterion = nn.CrossEntropyLoss()
    metrics, y_true, y_pred = _run_epoch(
        model, loader, criterion, device, optimizer=None
    )
    final_metrics = compute_classification_metrics(
        y_true, y_pred, class_names=class_names
    )
    final_metrics["loss"] = metrics["loss"]
    return final_metrics, y_true, y_pred


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    class_names=None,
    on_epoch=None,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = History()
    best_state = None
    best_val_f1 = -1.0
    last_val_true = None
    last_val_pred = None

    model.to(device)
    for epoch in range(1, epochs + 1):
        train_metrics, _, _ = _run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_metrics, y_true, y_pred = _run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )
        last_val_true, last_val_pred = y_true, y_pred

        history.train_loss.append(train_metrics["loss"])
        history.val_loss.append(val_metrics["loss"])
        history.train_acc.append(train_metrics["accuracy"])
        history.val_acc.append(val_metrics["accuracy"])

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if on_epoch is not None:
            on_epoch(epoch, train_metrics, val_metrics)

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics, last_val_true, last_val_pred = evaluate_model(
        model, val_loader, device, class_names=class_names
    )
    return history, final_metrics, last_val_true, last_val_pred
