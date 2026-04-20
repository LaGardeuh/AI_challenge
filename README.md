# Détection d'anomalies industrielles - PatchCore

POC de détection d'anomalies sur le dataset MVTec AD, dans le cadre du challenge IA ESAIP 2025-2026.

## Principe

On utilise **PatchCore** : un modèle non supervisé entraîné uniquement sur des images normales.
À l'inférence, on mesure la distance entre les features de l'image testée et la memory bank — plus la distance est grande, plus l'image est anormale.

Le backbone est un **WideResNet50** pré-entraîné sur ImageNet (couches layer2 + layer3 concaténées → 1536 features par patch).

## Installation

```bash
pip install -r poc/requirements.txt
```

## Utilisation

### 1. Entraîner et sauvegarder les modèles

À faire une seule fois. Nécessite le dataset MVTec AD.

```bash
python poc/save_models.py --data_root <chemin_vers_mvtec>
```

Les memory banks sont sauvegardées dans `models/` au format `.npy`.

### 2. Lancer l'application Gradio

```bash
python poc/app.py --models_dir ./models
```

Puis ouvrir **http://127.0.0.1:7860** dans le navigateur.

On sélectionne le type de composant dans le menu déroulant, on glisse-dépose une image, et le modèle affiche :
- le verdict (NORMAL / DEFAUT)
- le score d'anomalie
- la carte de chaleur superposée sur l'image

### 3. Évaluation complète (toutes les catégories)

```bash
python poc/main.py --data_root <chemin_vers_mvtec> --output_dir ./results
```

Génère les métriques (AUROC, F1, matrices de confusion) et les visualisations dans `results/`.

## Structure

```
poc/
├── app.py          # Interface Gradio
├── save_models.py  # Sauvegarde des memory banks
├── main.py         # Évaluation complète
├── model.py        # PatchCore + FeatureExtractor
├── dataset.py      # Chargement MVTec AD
├── evaluate.py     # Métriques (AUROC, F1, confusion)
├── visualize.py    # Heatmaps et graphiques
└── requirements.txt
models/             # Memory banks sauvegardées (.npy)
results/            # Résultats et visualisations
```

## Résultats

Évaluation sur les 15 catégories MVTec AD (train/good + test) :

| Métrique | Valeur |
|---|---|
| AUROC moyen | 0.971 |
| FN (défauts manqués) | 0 |
| FP (fausses alarmes) | 1327 |
| TP (défauts détectés) | 1258 |
