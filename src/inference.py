
import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm




def generate_anomaly_map(teacher, student, img_tensor: torch.Tensor,
                          img_size: int, device: str) -> np.ndarray:
    """
    Génère une heatmap d'anomalie pixel-level pour une image.

    Principe :
        - On compare les features Teacher et Student couche par couche.
        - L'erreur (MSE) est upsamplée à la taille de l'image.
        - On fusionne les deux échelles (layer2 + layer3).
        - Un filtre gaussien lisse le résultat.

    Args:
        teacher    : TeacherNetwork (gelé)
        student    : StudentNetwork (entraîné)
        img_tensor : [C, H, W] — image normalisée
        img_size   : taille cible de la sortie (ex: 256)
        device     : 'cuda' ou 'cpu'

    Returns:
        anomaly_map : np.ndarray [H, W] ∈ [0, 1]
    """
    teacher.eval()
    student.eval()

    img = img_tensor.unsqueeze(0).to(device)   # [1, C, H, W]

    with torch.no_grad():
        t_feats = teacher(img)
        s_feats = student(img)

    maps = []
    weights = {"feat_layer2": 0.4, "feat_layer3": 0.6}

    for key, w in weights.items():
        t = F.normalize(t_feats[key], dim=1)
        s = F.normalize(s_feats[key], dim=1)

        error = ((t - s) ** 2).mean(dim=1, keepdim=True)

        error = F.interpolate(
            error,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
        maps.append(w * error.squeeze().cpu().numpy())

    anomaly_map = sum(maps)

    anomaly_map = gaussian_filter(anomaly_map, sigma=4)

    anomaly_map = (anomaly_map - anomaly_map.min()) / \
                  (anomaly_map.max() - anomaly_map.min() + 1e-8)

    return anomaly_map


def compute_image_score(teacher, student, img_tensor: torch.Tensor,
                        device: str) -> float:
    """
    Score d'anomalie global pour une image entière.
    Utilisé pour l'évaluation image-level (AUROC).

    Score = max de l'anomaly map (pic d'anomalie le plus fort).
    """
    teacher.eval()
    student.eval()

    img = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        t_feats = teacher(img)
        s_feats = student(img)

    scores = []
    for key in ["feat_layer2", "feat_layer3"]:
        t = F.normalize(t_feats[key], dim=1)
        s = F.normalize(s_feats[key], dim=1)
        # Max spatial de l'erreur
        score = ((t - s) ** 2).mean(dim=1).max().item()
        scores.append(score)

    return float(np.mean(scores))



def _denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """Annule la normalisation ImageNet pour affichage."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)


def visualize_result(img_tensor: torch.Tensor,
                     anomaly_map: np.ndarray,
                     gt_mask: torch.Tensor | None,
                     save_path: str | None = None,
                     label: int = 0):
    """
    Affiche : image originale | anomaly map | superposition | GT mask.
    """
    img_np = _denormalize(img_tensor)
    h, w = anomaly_map.shape

    ncols = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    axes[0].imshow(img_np)
    axes[0].set_title("Image originale")
    axes[0].axis("off")

    axes[1].imshow(anomaly_map, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Anomaly Map")
    axes[1].axis("off")

    axes[2].imshow(img_np)
    axes[2].imshow(anomaly_map, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title(f"Superposition ({'défaut' if label else 'normal'})")
    axes[2].axis("off")

    if gt_mask is not None and ncols == 4:
        gt_np = gt_mask.squeeze().cpu().numpy()
        axes[3].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
        axes[3].set_title("Ground Truth")
        axes[3].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()



def run_inference(teacher, student, test_loader, cfg) -> dict:
    """
    Parcourt le test set, génère les anomaly maps et scores image-level.

    Returns:
        dict avec :
            'image_scores' : liste de floats (score par image)
            'image_labels' : liste de 0/1
            'anomaly_maps' : liste de np.ndarray [H, W]
            'gt_masks'     : liste de np.ndarray [H, W]
    """
    device = cfg.DEVICE
    os.makedirs(os.path.join(cfg.MAPS_DIR, cfg.CATEGORY), exist_ok=True)

    results = {
        "image_scores": [],
        "image_labels": [],
        "anomaly_maps": [],
        "gt_masks": [],
    }

    print(f"\n[Inference] Génération des anomaly maps — {cfg.CATEGORY}")

    for idx, (img, label, mask) in enumerate(tqdm(test_loader, ncols=80)):
        img_tensor = img.squeeze(0)
        label_int = label.item()
        gt_mask = mask.squeeze(0)

        score = compute_image_score(teacher, student, img_tensor, device)

        amap = generate_anomaly_map(
            teacher, student, img_tensor, cfg.IMG_SIZE, device
        )

        results["image_scores"].append(score)
        results["image_labels"].append(label_int)
        results["anomaly_maps"].append(amap)
        results["gt_masks"].append(gt_mask.squeeze().numpy())

        if idx % 20 == 0:
            save_path = os.path.join(
                cfg.MAPS_DIR, cfg.CATEGORY, f"result_{idx:04d}.png"
            )
            visualize_result(
                img_tensor, amap, gt_mask,
                save_path=save_path,
                label=label_int,
            )

    print(f"[Inference] {len(results['image_scores'])} images traitées")
    return results