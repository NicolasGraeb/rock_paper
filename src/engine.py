from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
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
    train_f1: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False


MONITORS = {
    "val_loss": ("loss", False),
    "val_f1_macro": ("f1_macro", True),
    "val_accuracy": ("accuracy", True),
}


def set_train_mode(model: nn.Module, train: bool, bn_eval: bool = False) -> None:
    """Jak model.train(train), ale BatchNorm zamrożonych warstw (albo wszystkie przy bn_eval) zostaje w eval."""
    model.train(train)
    if not train:
        return
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            frozen = not any(p.requires_grad for p in m.parameters(recurse=False))
            if bn_eval or frozen:
                m.eval()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str | None = None,
    on_batch=None,
    bn_eval: bool = False,
    return_probs: bool = False,
):
    train = optimizer is not None
    set_train_mode(model, train, bn_eval)
    loss_sum = 0.0
    preds_all = []
    targets_all = []
    probs_all = []
    n_correct = 0
    n_seen = 0

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, disable=desc is None)
    for inputs, targets in pbar:
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

        batch_loss = loss.item()
        loss_sum += batch_loss * targets.numel()
        preds = logits.argmax(dim=1)
        n_correct += (preds == targets).sum().item()
        n_seen += targets.numel()
        preds_all.append(preds.detach().cpu().numpy())
        targets_all.append(targets.detach().cpu().numpy())
        if return_probs:
            probs_all.append(torch.softmax(logits.detach().float(), dim=1).cpu().numpy())

        pbar.set_postfix(loss=f"{loss_sum / n_seen:.4f}", acc=f"{n_correct / n_seen:.4f}")
        if on_batch is not None:
            on_batch(batch_loss)

    y_pred = np.concatenate(preds_all)
    y_true = np.concatenate(targets_all)
    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["loss"] = loss_sum / n_seen
    if return_probs:
        metrics["probs"] = np.concatenate(probs_all)
    return metrics, y_true, y_pred


def compute_classification_metrics(y_true, y_pred, class_names=None):
    labels = list(range(len(class_names))) if class_names is not None else None
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
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
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": report,
    }
    return metrics


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names=None,
    return_probs: bool = False,
):
    criterion = nn.CrossEntropyLoss()
    metrics, y_true, y_pred = _run_epoch(
        model, loader, criterion, device, optimizer=None, return_probs=return_probs
    )
    final_metrics = compute_classification_metrics(
        y_true, y_pred, class_names=class_names
    )
    final_metrics["loss"] = metrics["loss"]
    if return_probs:
        final_metrics["probs"] = metrics["probs"]
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
    on_batch=None,
    monitor: str = "val_loss",
    patience: int | None = None,
    min_delta: float = 0.0,
    checkpoint_dir: str | Path | None = None,
    checkpoint_extra: dict | None = None,
    bn_eval: bool = False,
):
    metric_key, higher_is_better = MONITORS[monitor]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )
    history = History()
    best_state = None
    best_score = None
    epochs_without_improvement = 0
    last_val_true = None
    last_val_pred = None

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(path, epoch, val_metrics, with_optimizer):
        payload = {
            **(checkpoint_extra or {}),
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "monitor": monitor,
            "val_metrics": {k: v for k, v in val_metrics.items() if k != "classification_report"},
        }
        if with_optimizer:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(payload, path)

    model.to(device)
    epoch_bar = tqdm(range(1, epochs + 1), desc="epochs", dynamic_ncols=True)
    for epoch in epoch_bar:
        train_metrics, _, _ = _run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer,
            desc=f"train {epoch}/{epochs}", on_batch=on_batch, bn_eval=bn_eval,
        )
        val_metrics, y_true, y_pred = _run_epoch(
            model, val_loader, criterion, device, optimizer=None,
            desc=f"val   {epoch}/{epochs}",
        )
        last_val_true, last_val_pred = y_true, y_pred

        history.train_loss.append(train_metrics["loss"])
        history.val_loss.append(val_metrics["loss"])
        history.train_acc.append(train_metrics["accuracy"])
        history.val_acc.append(val_metrics["accuracy"])
        history.train_f1.append(train_metrics["f1_macro"])
        history.val_f1.append(val_metrics["f1_macro"])

        score = val_metrics[metric_key]
        if best_score is None:
            improved = True
        elif higher_is_better:
            improved = score > best_score + min_delta
        else:
            improved = score < best_score - min_delta

        if improved:
            best_score = score
            history.best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_dir is not None:
                save_checkpoint(checkpoint_dir / "best.pt", epoch, val_metrics, with_optimizer=False)
        else:
            epochs_without_improvement += 1

        if checkpoint_dir is not None:
            save_checkpoint(checkpoint_dir / "last.pt", epoch, val_metrics, with_optimizer=True)

        epoch_bar.set_postfix(
            train_loss=f"{train_metrics['loss']:.4f}",
            val_loss=f"{val_metrics['loss']:.4f}",
            val_acc=f"{val_metrics['accuracy']:.4f}",
            best=history.best_epoch,
        )
        if on_epoch is not None:
            on_epoch(epoch, train_metrics, val_metrics, history)

        if patience is not None and epochs_without_improvement >= patience:
            history.stopped_early = True
            tqdm.write(
                f"early stopping: {monitor} bez poprawy od {patience} epok "
                f"(najlepsza epoka {history.best_epoch})"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics, last_val_true, last_val_pred = evaluate_model(
        model, val_loader, device, class_names=class_names
    )
    return history, final_metrics, last_val_true, last_val_pred
