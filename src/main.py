import argparse
import os
import numpy as np
import joblib
from src.preprocessing import load_images_from_folder, normalize_images
from src.pca_model import FacePCAModel
from src.recognition import FaceRecognizer

def train(data_path, model_path, n_components=50):
    print(f"Chargement des images depuis {data_path}...")
    images, labels, _ = load_images_from_folder(data_path)
    
    if len(images) == 0:
        print("Erreur: Aucune image trouvée dans le dossier spécifié.")
        return

    print("Normalisation des données...")
    images_norm, mean, std = normalize_images(images)
    
    print(f"Entraînement de la PCA avec {n_components} composantes...")
    model = FacePCAModel(n_components=n_components)
    projected_features = model.train(images_norm)
    
    # Sauvegarde du modèle et des métadonnées
    model_data = {
        'model': model,
        'features': projected_features,
        'labels': labels,
        'mean': mean,
        'std': std
    }
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_data, model_path)
    print(f"Modèle sauvegardé avec succès dans {model_path}")

def identify(image_path, model_path):
    import cv2
    print(f"Chargement du modèle depuis {model_path}...")
    if not os.path.exists(model_path):
        print("Erreur: Le fichier modèle n'existe pas.")
        return

    model_data = joblib.load(model_path)
    model = model_data['model']
    trained_features = model_data['features']
    trained_labels = model_data['labels']
    mean = model_data['mean']
    std = model_data['std']

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Erreur: Impossible de lire l'image {image_path}")
        return

    img_resized = cv2.resize(img, (100, 100)).flatten()
    img_norm = (img_resized - mean) / std
    
    recognizer = FaceRecognizer(model, trained_features, trained_labels)
    label, distance = recognizer.identify(img_norm)
    
    print(f"Résultat de la reconnaissance : {label} (Distance: {distance:.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconnaissance faciale par PCA")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Entraîner le modèle")
    train_parser.add_argument("--data", default="data/raw", help="Dossier des images d'entraînement")
    train_parser.add_argument("--model", default="models/face_pca.pkl", help="Chemin de sauvegarde du modèle")
    train_parser.add_argument("--components", type=int, default=50, help="Nombre de composantes PCA")

    predict_parser = subparsers.add_parser("predict", help="Identifier un visage")
    predict_parser.add_argument("--image", required=True, help="Image à identifier")
    predict_parser.add_argument("--model", default="models/face_pca.pkl", help="Chemin du modèle entraîné")

    args = parser.parse_args()

    if args.command == "train":
        train(args.data, args.model, args.components)
    elif args.command == "predict":
        identify(args.image, args.model)
    else:
        parser.print_help()
