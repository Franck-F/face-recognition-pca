import cv2
import numpy as np
import os

def generate_synthetic_data(output_folder, num_people=5, images_per_person=10):
    """
    Génère des images synthétiques avec des motifs simples pour tester le pipeline.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print(f"Génération de {num_people * images_per_person} images dans {output_folder}...")
    
    for p in range(num_people):
        name = f"personne{p}"
        # Caractéristique de base pour cette personne (un motif unique)
        base_pattern = np.random.randint(0, 255, (20, 20), dtype=np.uint8)
        
        for i in range(images_per_person):
            # Création d'une image de 100x100
            img = np.zeros((100, 100), dtype=np.uint8)
            # Ajout du motif avec un peu de bruit
            noise = np.random.randint(-20, 20, (20, 20))
            pattern = np.clip(base_pattern.astype(int) + noise, 0, 255).astype(np.uint8)
            
            # Placement du motif au centre avec de légères variations de position
            off_x, off_y = np.random.randint(35, 45, 2)
            img[off_y:off_y+20, off_x:off_x+20] = pattern
            
            # Flou léger pour simuler une image réelle
            img = cv2.GaussianBlur(img, (5, 5), 0)
            
            filename = f"{name}_{i}.jpg"
            cv2.imwrite(os.path.join(output_folder, filename), img)

if __name__ == "__main__":
    generate_synthetic_data("data/raw")
