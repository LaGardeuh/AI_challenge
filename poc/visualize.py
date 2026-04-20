"""
Saves anomaly heatmaps overlaid on original images.
Useful for the report to show WHERE defects are detected.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_denorm = transforms.Normalize(
    mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
    std=[1 / s for s in IMAGENET_STD],
)


def _normalize_map(score_map: np.ndarray) -> np.ndarray:
    mn, mx = score_map.min(), score_map.max()
    if mx == mn:
        return np.zeros_like(score_map)
    return (score_map - mn) / (mx - mn)


def save_heatmaps(images, anomaly_maps, gt_masks, labels, defect_types,
                  output_dir: str, category: str, n_samples: int = 8):
    """
    Saves a grid of (original | heatmap | ground truth) for n_samples images.
    Picks examples with anomalies first.
    """
    out = Path(output_dir) / category
    out.mkdir(parents=True, exist_ok=True)

    anomaly_idxs = [i for i, l in enumerate(labels) if l == 1][:n_samples]
    good_idxs    = [i for i, l in enumerate(labels) if l == 0][:2]
    idxs = anomaly_idxs + good_idxs

    for idx in idxs:
        img_tensor = images[idx]
        img_np = _denorm(img_tensor).permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        score_map = _normalize_map(anomaly_maps[idx])
        gt_mask   = gt_masks[idx]
        label     = labels[idx]
        dtype     = defect_types[idx]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"{category} — {dtype} (label={'anomaly' if label else 'normal'})",
                     fontsize=12)

        axes[0].imshow(img_np)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(img_np)
        axes[1].imshow(score_map, cmap="jet", alpha=0.45)
        axes[1].set_title("Anomaly heatmap")
        axes[1].axis("off")

        axes[2].imshow(gt_mask, cmap="gray")
        axes[2].set_title("Ground truth mask")
        axes[2].axis("off")

        plt.tight_layout()
        fname = out / f"{dtype}_{idx:04d}.png"
        plt.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close()


def save_summary_chart(results: dict, output_dir: str):
    """Bar chart of image-level AUROC per category."""
    categories = list(results.keys())
    aurocs = [results[c]["image_auroc"] for c in categories]

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#2ecc71" if v >= 0.90 else "#e67e22" if v >= 0.75 else "#e74c3c"
              for v in aurocs]
    bars = ax.bar(categories, aurocs, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=np.nanmean(aurocs), color="navy", linestyle="--",
               label=f"Mean AUROC: {np.nanmean(aurocs):.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Image-level AUROC")
    ax.set_title("PatchCore — MVTec AD Results per Category")
    ax.legend()
    plt.xticks(rotation=30, ha="right")

    for bar, v in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(output_dir) / "summary_auroc.png", dpi=120)
    plt.close()
    print(f"\nSummary chart saved -> {output_dir}/summary_auroc.png")
