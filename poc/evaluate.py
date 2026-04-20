import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix


def _normalize(scores: np.ndarray) -> np.ndarray:
    mn, mx = scores.min(), scores.max()
    if mx == mn:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def image_level_metrics(scores: np.ndarray, labels: np.ndarray,
                        recall_target: float = None) -> dict:
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "f1": float("nan"),
                "precision": float("nan"), "recall": float("nan"),
                "threshold": float("nan"), "tp": 0, "tn": 0, "fp": 0, "fn": 0}

    scores_norm = _normalize(scores)
    auroc = roc_auc_score(labels, scores_norm)

    precisions, recalls, thresholds = precision_recall_curve(labels, scores_norm)
    f1_scores = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-8, None)

    if recall_target is not None:
        valid = np.where(recalls[:-1] >= recall_target)[0]
        best_idx = valid[-1] if len(valid) > 0 else 0
    else:
        best_idx = np.argmax(f1_scores[:-1])

    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]

    scores_bin = (scores_norm >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, scores_bin, labels=[0, 1]).ravel()

    return {
        "auroc": float(auroc),
        "f1": float(best_f1),
        "precision": float(best_precision),
        "recall": float(best_recall),
        "threshold": float(best_threshold),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def metrics_from_fixed_threshold(scores: np.ndarray, labels: np.ndarray,
                                  threshold: float) -> dict:
    from sklearn.metrics import roc_auc_score, confusion_matrix
    scores_norm = _normalize(scores)
    auroc = roc_auc_score(labels, scores_norm) if len(np.unique(labels)) > 1 else float("nan")

    scores_bin = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, scores_bin, labels=[0, 1]).ravel()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    return {
        "auroc": float(auroc), "f1": float(f1),
        "precision": float(prec), "recall": float(rec),
        "threshold": float(threshold),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def pixel_level_auroc(anomaly_maps: np.ndarray, gt_masks: np.ndarray) -> float:
    flat_scores = anomaly_maps.flatten()
    flat_labels = (gt_masks.flatten() > 0.5).astype(int)

    if flat_labels.sum() == 0:
        return float("nan")

    flat_scores = _normalize(flat_scores)
    return float(roc_auc_score(flat_labels, flat_scores))


def per_defect_metrics(scores: np.ndarray, labels: np.ndarray,
                       defect_types: list, recall_target: float = None) -> dict:
    unique_types = sorted(set(defect_types))
    results = {}
    for dt in unique_types:
        mask = np.array([d == dt or d == "good" for d in defect_types])
        sub_scores = scores[mask]
        sub_labels = labels[mask]
        if len(np.unique(sub_labels)) < 2:
            continue
        m = image_level_metrics(sub_scores, sub_labels, recall_target=recall_target)
        results[dt] = {"auroc": m["auroc"], "f1": m["f1"]}
    return results


def print_results(category: str, img_metrics: dict, pix_auroc: float):
    print(f"\n{'='*50}")
    print(f"  Categorie : {category.upper()}")
    print(f"{'='*50}")
    print(f"  AUROC image  : {img_metrics['auroc']:.4f}")
    print(f"  F1           : {img_metrics['f1']:.4f}")
    print(f"  Precision    : {img_metrics['precision']:.4f}")
    print(f"  Rappel       : {img_metrics['recall']:.4f}")
    print(f"  AUROC pixel  : {pix_auroc:.4f}")
    print(f"  Seuil        : {img_metrics['threshold']:.4f}")
    tp = img_metrics["tp"]
    tn = img_metrics["tn"]
    fp = img_metrics["fp"]
    fn = img_metrics["fn"]
    print(f"\n  Matrice de confusion :")
    print(f"               Predit:Normal  Predit:Defaut")
    print(f"  Reel:Normal  TN={tn:<6}    FP={fp}")
    print(f"  Reel:Defaut  FN={fn:<6}    TP={tp}")
