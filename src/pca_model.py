from sklearn.decomposition import PCA
import joblib
import os

class FacePCAModel:
    def __init__(self, n_components=50):
        """
        Initialise le modèle PCA.
        
        Args:
            n_components (int): Nombre de composantes principales à conserver.
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, whiten=True)
        self.is_trained = False

    def train(self, data):
        """
        Entraîne le modèle PCA sur les données d'images aplaties.
        """
        self.pca.fit(data)
        self.is_trained = True
        return self.pca.transform(data)

    def project(self, data):
        """
        Projette de nouvelles données dans l'espace des visages (eigenfaces).
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant la projection.")
        return self.pca.transform(data)

    def save(self, filepath):
        """
        Sauvegarde le modèle sur le disque.
        """
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath):
        """
        Charge un modèle depuis le disque.
        """
        return joblib.load(filepath)
