"""
PatchCore anomaly detection.

Training  : extract patch features from normal images -> build a coreset memory bank
Inference : nearest-neighbour distance to the memory bank = anomaly score
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm


class FeatureExtractor(nn.Module):
    """Extracts intermediate feature maps from WideResNet50 (layer2 + layer3)."""

    def __init__(self):
        super().__init__()
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        f2 = self.layer2(x)   # (B, 512, H/8, W/8)
        f3 = self.layer3(f2)  # (B, 1024, H/16, W/16)
        return f2, f3


def upsample_and_concat(f2, f3):
    """Upsample f3 to match f2 spatial size, then concatenate."""
    f3_up = nn.functional.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat([f2, f3_up], dim=1)  # (B, 1536, H, W)


def reshape_to_patches(features):
    """
    (B, C, H, W) -> (B*H*W, C)
    Returns patch embeddings and spatial dims.
    """
    B, C, H, W = features.shape
    patches = features.permute(0, 2, 3, 1).reshape(-1, C)
    return patches, H, W


def random_coreset(embeddings: np.ndarray, ratio: float = 0.1) -> np.ndarray:
    """Random subsampling — fast and nearly as good as greedy coreset."""
    n = len(embeddings)
    target = max(1, int(n * ratio))
    if target >= n:
        return embeddings
    idx = np.random.choice(n, target, replace=False)
    return embeddings[idx]


class PatchCore:
    def __init__(self, device: str = "cpu", coreset_ratio: float = 0.1, img_size: int = 224):
        self.device = device
        self.coreset_ratio = coreset_ratio
        self.img_size = img_size
        self.extractor = FeatureExtractor().to(device).eval()
        self.memory_bank: np.ndarray | None = None
        self.spatial_size: tuple | None = None  # (H, W) of feature map

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_dataset, batch_size: int = 8):
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        all_patches = []

        print("  Extracting features from training images...")
        with torch.no_grad():
            for images, *_ in tqdm(loader, leave=False):
                images = images.to(self.device)
                f2, f3 = self.extractor(images)
                feats = upsample_and_concat(f2, f3)
                patches, H, W = reshape_to_patches(feats)
                all_patches.append(patches.cpu().numpy())
                self.spatial_size = (H, W)

        all_patches = np.concatenate(all_patches, axis=0)
        print(f"  Total patches: {len(all_patches):,} -> applying coreset ({self.coreset_ratio:.0%})...")
        self.memory_bank = random_coreset(all_patches, ratio=self.coreset_ratio)
        print(f"  Memory bank size: {len(self.memory_bank):,} patches")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, test_dataset, batch_size: int = 4):
        """
        Returns:
            image_scores : (N,) anomaly score per image
            anomaly_maps : (N, img_size, img_size) pixel-level score maps
            labels       : (N,) ground truth labels
            masks        : (N, img_size, img_size) ground truth masks
        """
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        image_scores, anomaly_maps, labels, masks = [], [], [], []

        memory = torch.tensor(self.memory_bank, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            for images, gt_masks, gt_labels, _ in tqdm(loader, desc="  Inference", leave=False):
                images = images.to(self.device)
                f2, f3 = self.extractor(images)
                feats = upsample_and_concat(f2, f3)
                B, C, H, W = feats.shape

                patches = feats.permute(0, 2, 3, 1).reshape(B, H * W, C)

                for i in range(B):
                    p = patches[i]  # (H*W, C)
                    dists = torch.cdist(p, memory, p=2)        # (H*W, M)
                    min_dists, _ = dists.min(dim=1)            # (H*W,)

                    score_map = min_dists.reshape(H, W).cpu().numpy()
                    score_map_full = self._resize_map(score_map)

                    image_scores.append(float(score_map_full.max()))
                    anomaly_maps.append(score_map_full)

                labels.extend(gt_labels.numpy().tolist())
                for m in gt_masks:
                    masks.append(m.squeeze(0).numpy())

        return (
            np.array(image_scores),
            np.array(anomaly_maps),
            np.array(labels),
            np.array(masks),
        )

    def _resize_map(self, score_map: np.ndarray) -> np.ndarray:
        from PIL import Image as PILImage
        img = PILImage.fromarray(score_map.astype(np.float32))
        img = img.resize((self.img_size, self.img_size), PILImage.BILINEAR)
        return np.array(img)
