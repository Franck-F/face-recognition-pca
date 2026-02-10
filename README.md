# Face Recognition PCA

Ce projet implémente un système de reconnaissance faciale basé sur l'Analyse en Composantes Principales (PCA), utilisant la méthode des "Eigenfaces".

## Fonctionnalités

- Prétraitement automatisé des images (redimensionnement, niveaux de gris, normalisation).
- Réduction de dimensionnalité par PCA avec `scikit-learn`.
- Identification de visages par calcul de distance Euclidienne dans l'espace projeté.
- Interface en ligne de commande pour l'entraînement et l'inférence.
- Gestion des dépendances moderne avec `uv`.

## Installation

Assurez-vous d'avoir `uv` installé sur votre système.

```bash
uv sync
```

## Utilisation

### Structure des données

Placez vos images d'entraînement dans `data/raw/`. Le format de nom recommandé est `nom_id.jpg`.

### Entraînement

Pour entraîner le modèle :

```bash
uv run python src/main.py train --data data/raw --components 50
```

### Reconnaissance

Pour identifier un visage à partir d'une image :

```bash
uv run python src/main.py predict --image chemin/vers/image.jpg
```

## Architecture du Projet

- `src/` : Code source (prétraitement, modèle PCA, reconnaissance).
- `data/` : Dossiers pour les données brutes et traitées.
- `models/` : Stockage des modèles entraînés.
- `tests/` : Tests unitaires automatisés.

## Technologies Utilisées

- **Python**
- **OpenCV** (Traitement d'image)
- **Scikit-learn** (PCA & Machine Learning)
- **NumPy** (Algèbre linéaire)
- **uv** (Gestion de projet)

```
