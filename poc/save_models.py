"""
Script à lancer une seule fois pour entraîner et sauvegarder
les memory banks de toutes les catégories MVTec.
Les fichiers sont sauvegardés dans models/ au format .npy
"""

import argparse
import numpy as np
from pathlib import Path
from dataset import MVTecDataset, MVTEC_CATEGORIES
from model import PatchCore
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--models_dir", type=str, default="./models")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)

    for category in MVTEC_CATEGORIES:
        save_path = Path(args.models_dir) / f"{category}_memory_bank.npy"
        if save_path.exists():
            print(f"[{category}] deja sauvegarde, on passe")
            continue

        print(f"\n[{category}] Entrainement...")
        train_ds = MVTecDataset(args.data_root, category, split="train")
        model = PatchCore(device=device, coreset_ratio=0.1, knn=3, smooth_sigma=1.0)
        model.fit(train_ds)

        np.save(save_path, model.memory_bank)
        print(f"[{category}] Memory bank sauvegardee -> {save_path}")

    print("\nTous les modeles sont prets !")

if __name__ == "__main__":
    main()
