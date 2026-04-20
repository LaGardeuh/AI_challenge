
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm



def student_teacher_loss(teacher_feats: dict, student_feats: dict) -> torch.Tensor:
    """
    Calcule la loss MSE entre les features Teacher et Student.
    Normalisation L2 par canal avant la comparaison → stabilise l'entraînement.

    Args:
        teacher_feats : dict {'feat_layer2': tensor, 'feat_layer3': tensor}
        student_feats : dict {'feat_layer2': tensor, 'feat_layer3': tensor}

    Returns:
        loss scalaire
    """
    total_loss = torch.tensor(0.0, device=next(iter(teacher_feats.values())).device)

    for key in ["feat_layer2", "feat_layer3"]:
        t = F.normalize(teacher_feats[key], dim=1)   # [B, C, H, W]
        s = F.normalize(student_feats[key], dim=1)

        total_loss = total_loss + F.mse_loss(s, t)

    return total_loss



def train_student_teacher(teacher, student, train_loader, cfg):

    device = cfg.DEVICE

    teacher = teacher.to(device)
    student = student.to(device)

    optimizer = Adam(
        student.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)

    os.makedirs(cfg.WEIGHTS_DIR, exist_ok=True)
    best_loss = float("inf")
    best_path = os.path.join(cfg.WEIGHTS_DIR, f"student_{cfg.CATEGORY}.pth")

    print(f"\n{'='*60}")
    print(f"  Entraînement Student-Teacher: {cfg.CATEGORY}")
    print(f"  Epochs : {cfg.EPOCHS} | LR : {cfg.LR} | Device : {device}")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg.EPOCHS + 1):
        student.train()
        epoch_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{cfg.EPOCHS}",
            leave=False,
            ncols=80,
        )

        for imgs, _labels, _masks in progress:
            imgs = imgs.to(device)

            with torch.no_grad():
                teacher_feats = teacher(imgs)

            student_feats = student(imgs)

            loss = student_teacher_loss(teacher_feats, student_feats)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.6f}")

        scheduler.step()

        mean_loss = epoch_loss / len(train_loader)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d}/{cfg.EPOCHS} | Loss : {mean_loss:.6f} | LR : {scheduler.get_last_lr()[0]:.2e}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(student.state_dict(), best_path)

    print(f"\n Meilleur modèle sauvegardé → {best_path}")
    print(f" Loss minimale atteinte     : {best_loss:.6f}\n")

    student.load_state_dict(torch.load(best_path, map_location=device))
    return student



def load_student(student, cfg):
    """
    Charge les poids d'un student déjà entraîné depuis le dossier weights.

    Args:
        student : StudentNetwork instancié (architecture vide)
        cfg     : objet Config

    Returns:
        student avec les poids chargés
    """
    weights_path = os.path.join(cfg.WEIGHTS_DIR, f"student_{cfg.CATEGORY}.pth")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Aucun poids trouvé pour '{cfg.CATEGORY}' à : {weights_path}\n"
            f"Lance d'abord : python main.py --mode train"
        )

    student.load_state_dict(
        torch.load(weights_path, map_location=cfg.DEVICE)
    )
    print(f"[Train] Poids chargés depuis : {weights_path}")
    return student