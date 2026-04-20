

import json
import os
import numpy as np
from skimage import measure



def _binary_clf_curve(labels: np.ndarray, scores: np.ndarray):
    """
    Calcule TP, FP, TN, FN pour tous les seuils possibles.
    Implémentation from scratch sans sklearn.

    Returns:
        thresholds, tpr_array, fpr_array
    """
    sorted_idx  = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_idx]
    sorted_labels = labels[sorted_idx]

    thresholds = np.unique(sorted_scores)[::-1]

    tprs, fprs = [], []
    total_pos = labels.sum()
    total_neg = len(labels) - total_pos

    for thresh in thresholds:
        pred_pos = (scores >= thresh).astype(int)
        tp = ((pred_pos == 1) & (labels == 1)).sum()
        fp = ((pred_pos == 1) & (labels == 0)).sum()

        tpr = tp / (total_pos + 1e-8)
        fpr = fp / (total_neg + 1e-8)
        tprs.append(tpr)
        fprs.append(fpr)

    tprs = [0.0] + tprs + [1.0]
    fprs = [0.0] + fprs + [1.0]

    return np.array(thresholds), np.array(fprs), np.array(tprs)


def _auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    """Aire sous la courbe via méthode des trapèzes."""
    order = np.argsort(x)
    return float(np.trapezoid(y[order], x[order]))



def compute_image_auroc(image_labels: list, image_scores: list) -> float:
    """
    AUROC au niveau image : capacité à séparer images normales / défectueuses.

    Args:
        image_labels : liste de 0/1 (0=normal, 1=défaut)
        image_scores : liste de float (score d'anomalie par image)

    Returns:
        auroc ∈ [0, 1]
    """
    labels = np.array(image_labels)
    scores = np.array(image_scores)

    _, fprs, tprs = _binary_clf_curve(labels, scores)
    auroc = _auc_trapz(fprs, tprs)
    return auroc


def compute_pixel_auroc(gt_masks: list, anomaly_maps: list) -> float:
    """
    AUROC au niveau pixel : capacité à segmenter les pixels défectueux.

    Args:
        gt_masks     : liste de np.ndarray [H, W] binaires (0/1)
        anomaly_maps : liste de np.ndarray [H, W] ∈ [0, 1]

    Returns:
        auroc pixel ∈ [0, 1]
    """
    all_labels = np.concatenate([m.flatten() for m in gt_masks])
    all_scores = np.concatenate([a.flatten() for a in anomaly_maps])

    _, fprs, tprs = _binary_clf_curve(all_labels.astype(int), all_scores)
    return _auc_trapz(fprs, tprs)


def compute_au_pr(image_labels: list, image_scores: list) -> float:
    """
    Aire sous la courbe Precision-Recall (image-level).
    Adaptée aux classes déséquilibrées (peu de défauts vs beaucoup de normaux).
    """
    labels = np.array(image_labels)
    scores = np.array(image_scores)

    sorted_idx = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_idx]

    total_pos = labels.sum()
    precisions, recalls = [1.0], [0.0]

    tp, fp = 0, 0
    for lbl in sorted_labels:
        if lbl == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (total_pos + 1e-8)
        precisions.append(precision)
        recalls.append(recall)

    precisions.append(0.0)
    recalls.append(1.0)

    return _auc_trapz(np.array(recalls), np.array(precisions))



def compute_au_pro(gt_masks: list, anomaly_maps: list,
                   fpr_limit: float = 0.3,
                   n_thresholds: int = 100) -> float:
    """
    AU-PRO optimisé : précalcule les labels connectés une seule fois par image.
    200x plus rapide que la version naïve.
    """
    print("  [Pixel-level] Calcul AU-PRO en cours...", flush=True)

    precomputed = []
    for amap, gt in zip(anomaly_maps, gt_masks):
        gt_bin = (gt > 0.5).astype(np.uint8)
        neg_pixels = (gt_bin == 0).sum()

        labeled_gt = measure.label(gt_bin, connectivity=2)
        region_ids = np.unique(labeled_gt)[1:]

        regions = [labeled_gt == rid for rid in region_ids]
        region_sizes = [r.sum() for r in regions]

        precomputed.append({
            "amap": amap,
            "gt_bin": gt_bin,
            "neg_pixels": neg_pixels,
            "regions": regions,
            "region_sizes": region_sizes,
        })


    all_scores = np.concatenate([p["amap"].flatten() for p in precomputed])
    thresholds = np.percentile(all_scores, np.linspace(0, 100, n_thresholds))
    thresholds = np.unique(thresholds)[::-1]  # ordre décroissant

    pro_fpr_pairs = []

    for thresh in thresholds:
        all_pros, all_fprs = [], []

        for p in precomputed:
            pred = (p["amap"] >= thresh).astype(np.uint8)

            # FPR
            fp = ((pred == 1) & (p["gt_bin"] == 0)).sum()
            fpr = fp / (p["neg_pixels"] + 1e-8)
            all_fprs.append(fpr)

            if not p["regions"]:
                continue
            image_pros = [
                float((pred & region).sum() / (size + 1e-8))
                for region, size in zip(p["regions"], p["region_sizes"])
            ]
            all_pros.append(np.mean(image_pros))

        mean_fpr = float(np.mean(all_fprs))
        mean_pro = float(np.mean(all_pros)) if all_pros else 0.0

        if mean_fpr <= fpr_limit:
            pro_fpr_pairs.append((mean_fpr, mean_pro))

    if len(pro_fpr_pairs) < 2:
        return 0.0

    pro_fpr_pairs.sort(key=lambda x: x[0])
    fprs = np.array([p[0] for p in pro_fpr_pairs])
    pros = np.array([p[1] for p in pro_fpr_pairs])

    au_pro = _auc_trapz(fprs, pros) / (fpr_limit + 1e-8)
    return float(np.clip(au_pro, 0.0, 1.0))



def compute_all_metrics(results: dict, cfg) -> dict:
    """
    Calcule toutes les métriques à partir des résultats d'inférence.

    Args:
        results : dict retourné par inference.run_inference()
        cfg     : objet Config

    Returns:
        dict des métriques avec leurs valeurs
    """
    image_labels = results["image_labels"]
    image_scores = results["image_scores"]
    anomaly_maps = results["anomaly_maps"]
    gt_masks = results["gt_masks"]

    print(f"\n{'='*60}")
    print(f"  Évaluation — {cfg.CATEGORY}")
    print(f"{'='*60}")

    img_auroc = compute_image_auroc(image_labels, image_scores)
    img_aupr  = compute_au_pr(image_labels, image_scores)
    print(f"  [Image-level] AUROC : {img_auroc:.4f}")
    print(f"  [Image-level] AU-PR : {img_aupr:.4f}")

    has_gt = any(m.sum() > 0 for m in gt_masks)

    if has_gt:
        pix_auroc = compute_pixel_auroc(gt_masks, anomaly_maps)
        au_pro = compute_au_pro(gt_masks, anomaly_maps,
                                    fpr_limit=cfg.FPR_LIMIT)
        print(f"  [Pixel-level] AUROC  : {pix_auroc:.4f}")
        print(f"  [Pixel-level] AU-PRO : {au_pro:.4f}  (FPR ≤ {cfg.FPR_LIMIT})")
    else:
        pix_auroc = None
        au_pro = None
        print("  [Pixel-level] Aucun masque GT disponible.")

    print(f"{'='*60}\n")

    metrics = {
        "category": cfg.CATEGORY,
        "image_auroc": round(img_auroc, 4),
        "image_au_pr": round(img_aupr, 4),
        "pixel_auroc": round(pix_auroc, 4) if pix_auroc is not None else None,
        "au_pro": round(au_pro, 4)    if au_pro    is not None else None,
        "fpr_limit": cfg.FPR_LIMIT,
    }

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(cfg.RESULTS_DIR, f"{cfg.CATEGORY}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Métriques sauvegardées → {json_path}")

    return metrics