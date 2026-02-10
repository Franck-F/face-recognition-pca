import numpy as np
from scipy.spatial.distance import euclidean

class FaceRecognizer:
    def __init__(self, pca_model, trained_features, trained_labels):
        """
        Initialise le reconnaisseur avec les caractéristiques projetées et leurs étiquettes.
        """
        self.pca_model = pca_model
        self.trained_features = trained_features
        self.trained_labels = trained_labels

    def identify(self, test_image_flat):
        """
        Identifie une image de test en trouvant la distance minimale dans l'espace PCA.
        """
        # Projection de l'image de test
        test_projected = self.pca_model.project(test_image_flat.reshape(1, -1))
        
        distances = []
        for feature in self.trained_features:
            dist = euclidean(test_projected[0], feature)
            distances.append(dist)
            
        min_idx = np.argmin(distances)
        return self.trained_labels[min_idx], distances[min_idx]
