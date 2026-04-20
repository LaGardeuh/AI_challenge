"""
Custom evaluation — built from scratch (no MVTec eval code used).

Metrics:
  - Image-level AUROC  : overall detection ability
  - Image-level F1     : with optimal threshold search
  - Pixel-level AUROC  : localisation quality
  - Precision / Recall : at optimal threshold
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc


def _normalize(scores: np.ndarray) -> np.ndarray:
    mn, mx = scores.min(), scores.max()
    if mx == mn:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def image_level_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """
    scores : (N,) anomaly score per image (higher = more anomalous)
    labels : (N,) 0=normal, 1=anomaly
    """
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "f1": float("nan"),
                "precision": float("nan"), "recall": float("nan"),
                "threshold": float("nan")}

    scores_norm = _normalize(scores)
    auroc = roc_auc_score(labels, scores_norm)

    # Find threshold that maximises F1
    precisions, recalls, thresholds = precision_recall_curve(labels, scores_norm)
    f1_scores = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-8, None)
    best_idx = np.argmax(f1_scores[:-1])  # last element has no threshold
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]

    return {
        "auroc": float(auroc),
        "f1": float(best_f1),
        "precision": float(best_precision),
        "recall": float(best_recall),
        "threshold": float(best_threshold),
    }


def pixel_level_auroc(anomaly_maps: np.ndarray, gt_masks: np.ndarray) -> float:
    """
    anomaly_maps : (N, H, W) float score maps
    gt_masks     : (N, H, W) binary ground truth (0/1)
    Returns pixel-level AUROC.
    """
    flat_scores = anomaly_maps.flatten()
    flat_labels = (gt_masks.flatten() > 0.5).astype(int)

    if flat_labels.sum() == 0:
        return float("nan")

    flat_scores = _normalize(flat_scores)
    return float(roc_auc_score(flat_labels, flat_scores))


def per_defect_metrics(scores: np.ndarray, labels: np.ndarray,
                       defect_types: list) -> dict:
    """
    Returns F1 per defect type (for the report).
    """
    unique_types = sorted(set(defect_types))
    results = {}
    for dt in unique_types:
        mask = np.array([d == dt or d == "good" for d in defect_types])
        sub_scores = scores[mask]
        sub_labels = labels[mask]
        if len(np.unique(sub_labels)) < 2:
            continue
        m = image_level_metrics(sub_scores, sub_labels)
        results[dt] = {"auroc": m["auroc"], "f1": m["f1"]}
    return results


def print_results(category: str, img_metrics: dict, pix_auroc: float):
    print(f"\n{'='*50}")
    print(f"  Category : {category.upper()}")
    print(f"{'='*50}")
    print(f"  Image AUROC   : {img_metrics['auroc']:.4f}")
    print(f"  Image F1      : {img_metrics['f1']:.4f}")
    print(f"  Precision     : {img_metrics['precision']:.4f}")
    print(f"  Recall        : {img_metrics['recall']:.4f}")
    print(f"  Pixel AUROC   : {pix_auroc:.4f}")
    print(f"  Threshold     : {img_metrics['threshold']:.4f}")
