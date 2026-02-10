import cv2
import numpy as np
import os

def load_images_from_folder(folder, size=(100, 100)):
    """
    Charge les images d'un dossier, les convertit en niveaux de gris et les redimensionne.
    
    Args:
        folder (str): Chemin vers le dossier contenant les images.
        size (tuple): Taille cible pour le redimensionnement (largeur, hauteur).
        
    Returns:
        tuple: (Données des images aplaties, étiquettes, noms des fichiers)
    """
    images = []
    labels = []
    filenames = []
    
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            img_resized = cv2.resize(img, size)
            images.append(img_resized.flatten())
            # On suppose que le format du nom est 'label_id.jpg' ou similaire
            label = filename.split('_')[0]
            labels.append(label)
            filenames.append(filename)
            
    return np.array(images), labels, filenames

def normalize_images(images):
    """
    Normalise les données d'image pour avoir une moyenne de 0 et un écart-type de 1.
    """
    mean = np.mean(images, axis=0)
    std = np.std(images, axis=0)
    # Éviter la division par zéro
    std[std == 0] = 1.0
    return (images - mean) / std, mean, std
