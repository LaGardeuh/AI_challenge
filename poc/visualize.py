"""
Génération des visualisations pour le rapport :
- Heatmaps d'anomalie superposées sur les images originales
- Matrices de confusion par catégorie et globale
- Graphique bilan de l'AUROC par catégorie
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# transformation inverse pour récupérer l'image originale depuis le tenseur normalisé
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
    Sauvegarde des images côte à côte : original | heatmap | masque ground truth.
    On prend en priorité des images avec défauts pour que ce soit plus parlant.
    """
    out = Path(output_dir) / category
    out.mkdir(parents=True, exist_ok=True)

    # on sélectionne d'abord les anomalies puis quelques images normales
    anomaly_idxs = [i for i, l in enumerate(labels) if l == 1][:n_samples]
    good_idxs = [i for i, l in enumerate(labels) if l == 0][:2]
    idxs = anomaly_idxs + good_idxs

    for idx in idxs:
        img_tensor = images[idx]
        img_np = _denorm(img_tensor).permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        score_map = _normalize_map(anomaly_maps[idx])
        gt_mask = gt_masks[idx]
        label = labels[idx]
        dtype = defect_types[idx]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"{category} - {dtype} ({'defaut' if label else 'normal'})", fontsize=12)

        axes[0].imshow(img_np)
        axes[0].set_title("Image originale")
        axes[0].axis("off")

        # la heatmap est superposée en transparence sur l'image originale
        axes[1].imshow(img_np)
        axes[1].imshow(score_map, cmap="jet", alpha=0.45)
        axes[1].set_title("Carte d'anomalie")
        axes[1].axis("off")

        axes[2].imshow(gt_mask, cmap="gray")
        axes[2].set_title("Masque ground truth")
        axes[2].axis("off")

        plt.tight_layout()
        fname = out / f"{dtype}_{idx:04d}.png"
        plt.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close()


def save_confusion_matrix(img_metrics: dict, category: str, output_dir: str):
    """
    Matrice de confusion visuelle avec annotations métier.
    FP = bonne pièce rejetée (perte argent), FN = défaut non détecté (perte réputation).
    """
    tp = img_metrics["tp"]
    tn = img_metrics["tn"]
    fp = img_metrics["fp"]
    fn = img_metrics["fn"]

    matrix = np.array([[tn, fp], [fn, tp]])
    cell_labels = np.array([["TN", "FP\n(perte argent)"], ["FN\n(perte reputation)", "TP"]])
    # vert pour les bonnes prédictions, rouge pour les erreurs
    colors = np.array([[0.2, 0.8], [0.8, 0.2]])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(colors, cmap=plt.cm.RdYlGn, vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predit: Normal", "Predit: Defaut"])
    ax.set_yticklabels(["Reel: Normal", "Reel: Defaut"])
    ax.set_title(f"Matrice de confusion - {category}")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cell_labels[i, j]}\n{matrix[i, j]}",
                    ha="center", va="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    out = Path(output_dir) / "confusion_matrices"
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / f"{category}_confusion.png", dpi=120, bbox_inches="tight")
    plt.close()


def save_global_confusion_matrix(results: dict, output_dir: str):
    """Matrice de confusion agrégée sur toutes les catégories."""
    tp = sum(r["tp"] for r in results.values() if "tp" in r)
    tn = sum(r["tn"] for r in results.values() if "tn" in r)
    fp = sum(r["fp"] for r in results.values() if "fp" in r)
    fn = sum(r["fn"] for r in results.values() if "fn" in r)

    matrix = np.array([[tn, fp], [fn, tp]])
    cell_labels = np.array([["TN", "FP\n(perte argent)"], ["FN\n(perte reputation)", "TP"]])
    colors = np.array([[0.2, 0.8], [0.8, 0.2]])

    _, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(colors, cmap=plt.cm.RdYlGn, vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predit: Normal", "Predit: Defaut"])
    ax.set_yticklabels(["Reel: Normal", "Reel: Defaut"])
    ax.set_title("Matrice de confusion globale - toutes categories")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cell_labels[i, j]}\n{matrix[i, j]}",
                    ha="center", va="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(output_dir) / "global_confusion.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Matrice de confusion globale sauvegardee -> {output_dir}/global_confusion.png")


def save_summary_chart(results: dict, output_dir: str):
    """Graphique barre de l'AUROC par catégorie avec la moyenne."""
    categories = list(results.keys())
    aurocs = [results[c]["image_auroc"] for c in categories]

    fig, ax = plt.subplots(figsize=(14, 5))
    # code couleur : vert si > 0.9, orange si > 0.75, rouge sinon
    colors = ["#2ecc71" if v >= 0.90 else "#e67e22" if v >= 0.75 else "#e74c3c"
              for v in aurocs]
    bars = ax.bar(categories, aurocs, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=np.nanmean(aurocs), color="navy", linestyle="--",
               label=f"Moyenne AUROC: {np.nanmean(aurocs):.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC image")
    ax.set_title("PatchCore - Resultats MVTec AD par categorie")
    ax.legend()
    plt.xticks(rotation=30, ha="right")

    for bar, v in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(output_dir) / "summary_auroc.png", dpi=120)
    plt.close()
    print(f"\nGraphique AUROC sauvegarde -> {output_dir}/summary_auroc.png")
