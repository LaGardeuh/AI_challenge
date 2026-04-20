import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_mask_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])


class MVTecDataset(Dataset):

    def __init__(self, root: str, category: str, split: str, img_size: int = 224):
        self.root = Path(root) / category
        self.split = split
        self.transform = get_transforms(img_size)
        self.mask_transform = get_mask_transforms(img_size)
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        split_dir = self.root / self.split
        for defect_type in sorted(os.listdir(split_dir)):
            defect_dir = split_dir / defect_type
            if not defect_dir.is_dir():
                continue
            label = 0 if defect_type == "good" else 1
            for img_path in sorted(defect_dir.glob("*.png")):
                if label == 1:
                    mask_path = (
                        self.root / "ground_truth" / defect_type /
                        (img_path.stem + "_mask.png")
                    )
                else:
                    mask_path = None
                self.samples.append((img_path, mask_path, label, defect_type))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label, defect_type = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        if mask_path and mask_path.exists():
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask)
        else:
            import torch
            mask = torch.zeros(1, image.shape[1], image.shape[2])

        return image, mask, label, defect_type
