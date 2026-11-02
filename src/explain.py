import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def gradcam_layer(model: nn.Module) -> nn.Module:
    return model.features[-1]


def grad_cam(model: nn.Module, x: torch.Tensor, class_idx: torch.Tensor | None = None):
    model.eval()
    acts = {}
    handle = gradcam_layer(model).register_forward_hook(lambda m, i, o: acts.update(a=o))
    try:
        x = x.clone().requires_grad_(True)
        logits = model(x)
        idx = logits.argmax(1) if class_idx is None else class_idx
        grads = torch.autograd.grad(logits.gather(1, idx[:, None]).sum(), acts["a"])[0]
    finally:
        handle.remove()
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam - cam.amin(dim=(1, 2), keepdim=True)
    cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return cam.detach().cpu().numpy(), logits.detach()


@torch.no_grad()
def occlusion_sensitivity(model: nn.Module, x: torch.Tensor, class_idx: int, patch: int = 32, stride: int = 16,
                          batch_size: int = 64) -> np.ndarray:
    """Spadek p(class_idx) po zasłonięciu kwadratu patch x patch (zasłona = średni kolor)."""
    model.eval()
    _, _, h, w = x.shape
    base = F.softmax(model(x), 1)[0, class_idx].item()
    positions = [(t, l) for t in range(0, h - patch + 1, stride) for l in range(0, w - patch + 1, stride)]
    drops = []
    for i in range(0, len(positions), batch_size):
        chunk = positions[i:i + batch_size]
        xb = x.repeat(len(chunk), 1, 1, 1)
        for j, (t, l) in enumerate(chunk):
            xb[j, :, t:t + patch, l:l + patch] = 0
        drops.append(base - F.softmax(model(xb), 1)[:, class_idx].cpu())
    drops = torch.cat(drops)
    heat, count = torch.zeros(h, w), torch.zeros(h, w)
    for d, (t, l) in zip(drops, positions):
        heat[t:t + patch, l:l + patch] += d
        count[t:t + patch, l:l + patch] += 1
    return (heat / count.clamp_min(1)).numpy()


def plot_side_by_side(rows: list[dict], out_path, title: str, maps: tuple[str, ...] = ("gradcam",)):
    cols = 1 + len(maps)
    fig, axes = plt.subplots(len(rows), cols, figsize=(3 * cols, 3 * len(rows)), squeeze=False)
    for ax_row, r in zip(axes, rows):
        ok = r["true"] == r["pred"]
        ax_row[0].imshow(r["image"])
        ax_row[0].set_title(f"prawda: {r['true']}\npredykcja: {r['pred']} ({r['confidence']:.0%})", fontsize=8,
                            color="black" if ok else "tab:red")
        for ax, name in zip(ax_row[1:], maps):
            ax.imshow(r["image"])
            if name == "gradcam":
                ax.imshow(r["gradcam"], cmap="jet", alpha=0.45, vmin=0, vmax=1)
                ax.set_title(f"Grad-CAM ({r['pred']})", fontsize=8)
            else:
                lim = max(abs(r["occlusion"]).max(), 1e-6)
                ax.imshow(r["occlusion"], cmap="RdBu_r", alpha=0.55, vmin=-lim, vmax=lim)
                ax.set_title(f"occlusion: spadek p({r['true']})\nmax {lim:.2f}", fontsize=8)
        for ax in ax_row:
            ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (3 * len(rows))))
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
