"""
Lancement :
    python app.py --models_dir ./models --threshold 0.5
"""

import argparse
import numpy as np
import torch
import gradio as gr
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from model import FeatureExtractor, upsample_and_concat
from dataset import MVTEC_CATEGORIES, IMAGENET_MEAN, IMAGENET_STD
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

denorm = transforms.Normalize(
    mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
    std=[1 / s for s in IMAGENET_STD],
)

def load_memory_banks(models_dir: str) -> dict:
    banks = {}
    for cat in MVTEC_CATEGORIES:
        path = Path(models_dir) / f"{cat}_memory_bank.npy"
        if path.exists():
            banks[cat] = np.load(str(path))
            print(f"  Charge: {cat} ({len(banks[cat]):,} patches)")
        else:
            print(f"  Manquant: {cat}")
    return banks


def compute_anomaly_score(image_pil: Image.Image, memory_bank: np.ndarray,
                          extractor: FeatureExtractor, device: str,
                          knn: int = 3, smooth_sigma: float = 1.0) -> tuple:
    img_tensor = transform(image_pil.convert("RGB")).unsqueeze(0).to(device)

    memory = torch.tensor(memory_bank, dtype=torch.float32).to(device)

    with torch.no_grad():
        f2, f3 = extractor(img_tensor)
        feats = upsample_and_concat(f2, f3)
        B, C, H, W = feats.shape
        patches = feats.permute(0, 2, 3, 1).reshape(B, H * W, C)

        p = patches[0]
        dists = torch.cdist(p, memory, p=2)
        knn_dists = dists.topk(knn, largest=False).values
        patch_scores = knn_dists.mean(dim=1)

        score_map = patch_scores.reshape(H, W).cpu().numpy()
        score_map = gaussian_filter(score_map, sigma=smooth_sigma)

        score_map_pil = Image.fromarray(score_map.astype(np.float32))
        score_map_full = np.array(score_map_pil.resize((224, 224), Image.BILINEAR))

    score = float(np.percentile(score_map_full, 99))

    mn, mx = score_map_full.min(), score_map_full.max()
    score_map_norm = (score_map_full - mn) / (mx - mn + 1e-8)

    img_np = denorm(img_tensor.squeeze(0).cpu()).permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)

    return score, score_map_norm, img_np


def build_result_image(img_np: np.ndarray, score_map: np.ndarray,
                       verdict: str, score: float) -> np.ndarray:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    color = "#e74c3c" if verdict == "DEFAUT" else "#2ecc71"
    fig.suptitle(f"{verdict}  |  Score: {score:.4f}", fontsize=14,
                 fontweight="bold", color=color)

    axes[0].imshow(img_np)
    axes[0].set_title("Image originale")
    axes[0].axis("off")

    axes[1].imshow(img_np)
    axes[1].imshow(score_map, cmap="jet", alpha=0.5)
    axes[1].set_title("Carte d'anomalie")
    axes[1].axis("off")

    plt.tight_layout()
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    result = np.array(Image.open(buf).convert("RGB"))
    plt.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, default="./models")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\nChargement des memory banks...")
    memory_banks = load_memory_banks(args.models_dir)

    if not memory_banks:
        print("Aucun modele trouve.")
        return

    print("\nChargement du feature extractor...")
    extractor = FeatureExtractor().to(device).eval()

    thresholds = {}

    available_categories = list(memory_banks.keys())

    def predict(image, category):
        if image is None:
            return None, "Veuillez charger une image."

        if category not in memory_banks:
            return None, f"Modele non disponible pour '{category}'."

        try:
            image_pil = Image.fromarray(image)
            score, score_map, img_np = compute_anomaly_score(
                image_pil, memory_banks[category], extractor, device
            )

            threshold = args.threshold if args.threshold else thresholds.get(category, 0.5)

            verdict = "DEFAUT" if score > threshold else "NORMAL"
            result_img = build_result_image(img_np, score_map, verdict, score)

            info = (
                f"Categorie : {category}\n"
                f"Score     : {score:.6f}\n"
                f"Seuil     : {threshold:.6f}\n"
                f"Verdict   : {verdict}"
            )
            return result_img, info
        except Exception as e:
            import traceback
            return None, f"ERREUR:\n{traceback.format_exc()}"

    with gr.Blocks(title="Détection d'anomalies industrielles") as demo:
        gr.Markdown("## Détection d'anomalies industrielles - PatchCore")
        gr.Markdown("Sélectionnez le type de composant, puis glissez-déposez une image.")

        with gr.Row():
            with gr.Column():
                category_input = gr.Dropdown(
                    choices=available_categories,
                    label="Type de composant",
                    value=available_categories[0],
                )
                image_input = gr.Image(
                    label="Image à analyser (glisser-déposer)",
                    type="numpy",
                )
                submit_btn = gr.Button("Analyser", variant="primary")

            with gr.Column():
                result_image = gr.Image(label="Résultat")
                result_text = gr.Textbox(label="Détails", lines=5)

        submit_btn.click(
            fn=predict,
            inputs=[image_input, category_input],
            outputs=[result_image, result_text],
        )

        image_input.change(
            fn=predict,
            inputs=[image_input, category_input],
            outputs=[result_image, result_text],
        )

    print("\nLancement de l'application...")
    demo.launch(share=False)


if __name__ == "__main__":
    main()
