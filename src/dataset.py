
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class MVTecDataset(Dataset):
    """
    Dataset MVTec AD pour le Student-Teacher.
    - split='train' : uniquement les images 'good' (unsupervised)
    - split='test'  : images good + défectueuses, avec masques GT
    """

    def __init__(self, root: str, category: str, split: str = "train", transform=None):
        self.split = split
        self.transform = transform
        self.samples = []
        self.masks = []

        base = Path(root) / category

        if split == "train":
            good_dir = base / "train" / "good"
            for img_path in sorted(good_dir.glob("*.png")):
                self.samples.append((img_path, 0))
                self.masks.append(None)

        elif split == "test":
            test_dir = base / "test"
            gt_dir = base / "ground_truth"

            for defect_dir in sorted(test_dir.iterdir()):
                label = 0 if defect_dir.name == "good" else 1
                for img_path in sorted(defect_dir.glob("*.png")):
                    self.samples.append((img_path, label))

                    if label == 1:
                        mask_path = gt_dir / defect_dir.name / (img_path.stem + "_mask.png")
                        self.masks.append(mask_path if mask_path.exists() else None)
                    else:
                        self.masks.append(None)
        else:
            raise ValueError(f"split doit être 'train' ou 'test', reçu : {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        mask_path = self.masks[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")
            mask = T.Resize((img.shape[-2], img.shape[-1]),
                            interpolation=T.InterpolationMode.NEAREST)(mask)
            mask = T.ToTensor()(mask)
            mask = (mask > 0.5).float()
        else:
            h, w = img.shape[-2], img.shape[-1]
            mask = torch.zeros(1, h, w)

        return img, label, mask


# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(img_size: int):
    """Retourne les transforms train et test."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    test_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return train_transform, test_transform


def get_dataloaders(cfg, mode: str = "all"):
    """
    Construit et retourne les DataLoaders train et/ou test.

    Args:
        cfg  : objet Config (voir config.py)
        mode : 'train' | 'eval' | 'all'

    Returns:
        train_loader (None si mode='eval'),
        test_loader  (None si mode='train')
    """
    data_path = Path(cfg.DATA_ROOT) / cfg.CATEGORY
    if not data_path.exists():
        raise FileNotFoundError(
            f"\n\n  ✗ Dataset introuvable : {data_path.resolve()}\n"
            f"  → Vérifie DATA_ROOT dans config.py\n"
            f"  → Valeur actuelle : '{cfg.DATA_ROOT}'\n"
        )

    train_transform, test_transform = get_transforms(cfg.IMG_SIZE)
    train_loader = None
    test_loader = None

    if mode in ["train", "all"]:
        train_dataset = MVTecDataset(
            root=cfg.DATA_ROOT,
            category=cfg.CATEGORY,
            split="train",
            transform=train_transform,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        print(f"[Dataset] Catégorie     : {cfg.CATEGORY}")
        print(f"[Dataset] Train samples : {len(train_dataset)}")

    # ── Test loader ──────────────────────────────────────────────────────────
    if mode in ["eval", "all"]:
        test_dataset = MVTecDataset(
            root=cfg.DATA_ROOT,
            category=cfg.CATEGORY,
            split="test",
            transform=test_transform,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        print(f"[Dataset] Test  samples : {len(test_dataset)}")

    return train_loader, test_loader