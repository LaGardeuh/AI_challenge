# main.py

import argparse
import torch
from config import Config
from src.dataset import get_dataloaders
from src.model import TeacherNetwork, StudentNetwork, build_models
from src.train import train_student_teacher, load_student
from src.inference import run_inference
from src.metrics import compute_all_metrics


def main():
    parser = argparse.ArgumentParser(description="Student-Teacher Anomaly Detection — MVTec AD")
    parser.add_argument(
        "--category",
        type=str,
        default=Config.CATEGORY,
        help=f"Catégorie MVTec (défaut: {Config.CATEGORY})",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "all"],
        default="all",
        help="train=entraînement seul | eval=évaluation seul | all=les deux",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────────
    cfg          = Config()
    cfg.CATEGORY = args.category
    cfg.DEVICE   = args.device

    print(f"\n  Device : {cfg.DEVICE}")
    print(f"  Mode   : {args.mode}")

    # ── Dataloaders ──────────────────────────────────────────────────────────
    train_loader, test_loader = get_dataloaders(cfg, mode=args.mode)

    # ── Modèles ──────────────────────────────────────────────────────────────
    teacher, student = build_models(cfg.DEVICE)

    # ── Train ────────────────────────────────────────────────────────────────
    if args.mode in ["train", "all"]:
        student = train_student_teacher(teacher, student, train_loader, cfg)

    # ── Eval ─────────────────────────────────────────────────────────────────
    if args.mode in ["eval", "all"]:
        if args.mode == "eval":
            # Charger les poids existants si on fait eval seul
            student = load_student(student, cfg)

        results = run_inference(teacher, student, test_loader, cfg)
        compute_all_metrics(results, cfg)


if __name__ == "__main__":
    main()