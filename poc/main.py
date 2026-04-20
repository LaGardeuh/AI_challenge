"""
AI Challenge — Industrial Defect Detection POC
PatchCore on MVTec AD (all 15 categories)

Usage:
    python main.py --data_root ../mvtec_data --output_dir ./results
    python main.py --data_root ../mvtec_data --category bottle  # single category
"""

import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import MVTecDataset, MVTEC_CATEGORIES
from model import PatchCore
from evaluate import image_level_metrics, pixel_level_auroc, per_defect_metrics, print_results
from visualize import save_heatmaps, save_summary_chart


def run_category(category: str, data_root: str, output_dir: str,
                 device: str, coreset_ratio: float, img_size: int,
                 save_viz: bool) -> dict:

    print(f"\n{'#'*60}")
    print(f"  Processing: {category.upper()}")
    print(f"{'#'*60}")
    t0 = time.time()

    train_ds = MVTecDataset(data_root, category, split="train", img_size=img_size)
    test_ds  = MVTecDataset(data_root, category, split="test",  img_size=img_size)
    print(f"  Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")

    model = PatchCore(device=device, coreset_ratio=coreset_ratio, img_size=img_size)
    model.fit(train_ds)

    image_scores, anomaly_maps, labels, masks = model.predict(test_ds)

    defect_types = [test_ds.samples[i][3] for i in range(len(test_ds))]

    img_metrics  = image_level_metrics(image_scores, labels)
    pix_auroc    = pixel_level_auroc(anomaly_maps, masks)
    defect_breakdown = per_defect_metrics(image_scores, labels, defect_types)

    print_results(category, img_metrics, pix_auroc)

    if save_viz:
        # Collect raw images for visualisation
        images_list = [test_ds[i][0] for i in range(len(test_ds))]
        import torch as _torch
        images_tensor = _torch.stack(images_list)
        save_heatmaps(
            images_tensor, anomaly_maps, masks, labels, defect_types,
            output_dir=str(Path(output_dir) / "heatmaps"),
            category=category,
        )

    elapsed = time.time() - t0
    result = {
        "category": category,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "image_auroc": img_metrics["auroc"],
        "image_f1": img_metrics["f1"],
        "precision": img_metrics["precision"],
        "recall": img_metrics["recall"],
        "pixel_auroc": pix_auroc,
        "threshold": img_metrics["threshold"],
        "defect_breakdown": defect_breakdown,
        "elapsed_s": round(elapsed, 1),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="PatchCore — MVTec AD POC")
    parser.add_argument("--data_root",     type=str, required=True,
                        help="Path to extracted MVTec AD dataset root")
    parser.add_argument("--output_dir",    type=str, default="./results")
    parser.add_argument("--category",      type=str, default=None,
                        help="Single category to run (default: all 15)")
    parser.add_argument("--coreset_ratio", type=float, default=0.1,
                        help="Fraction of patches to keep in memory bank (default: 0.1)")
    parser.add_argument("--img_size",      type=int, default=224)
    parser.add_argument("--no_viz",        action="store_true",
                        help="Skip heatmap generation (faster)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Data root: {args.data_root}")
    print(f"Coreset ratio: {args.coreset_ratio}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    categories = [args.category] if args.category else MVTEC_CATEGORIES
    all_results = {}

    for cat in categories:
        result = run_category(
            category=cat,
            data_root=args.data_root,
            output_dir=args.output_dir,
            device=device,
            coreset_ratio=args.coreset_ratio,
            img_size=args.img_size,
            save_viz=not args.no_viz,
        )
        all_results[cat] = result

    # ---- Summary ----
    valid = [r for r in all_results.values() if not np.isnan(r["image_auroc"])]
    mean_img_auroc = np.mean([r["image_auroc"] for r in valid])
    mean_pix_auroc = np.mean([r["pixel_auroc"] for r in valid
                               if not np.isnan(r["pixel_auroc"])])
    mean_f1        = np.mean([r["image_f1"] for r in valid])

    print(f"\n{'='*60}")
    print(f"  GLOBAL SUMMARY ({len(valid)}/{len(categories)} categories)")
    print(f"{'='*60}")
    print(f"  Mean Image AUROC : {mean_img_auroc:.4f}")
    print(f"  Mean Pixel AUROC : {mean_pix_auroc:.4f}")
    print(f"  Mean F1          : {mean_f1:.4f}")

    # Save JSON results
    results_path = Path(args.output_dir) / "results.json"
    summary = {
        "mean_image_auroc": round(mean_img_auroc, 4),
        "mean_pixel_auroc": round(mean_pix_auroc, 4),
        "mean_f1": round(mean_f1, 4),
        "categories": all_results,
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved -> {results_path}")

    if not args.no_viz:
        save_summary_chart(all_results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
