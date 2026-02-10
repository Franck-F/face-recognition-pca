import numpy as np
import pytest
from src.preprocessing import normalize_images
from src.pca_model import FacePCAModel

def test_normalize_images():
    # Création de données bidimensionnelles factices
    data = np.array([[10, 20], [30, 40], [50, 60]], dtype=float)
    norm_data, mean, std = normalize_images(data)
    
    # Vérifier que la moyenne est proche de 0 après normalisation
    assert np.allclose(np.mean(norm_data, axis=0), 0)
    # Vérifier que l'écart-type est proche de 1
    assert np.allclose(np.std(norm_data, axis=0), 1)

def test_pca_model():
    # Création de données factices (10 échantillons, 100 caractéristiques)
    data = np.random.rand(10, 100)
    n_components = 5
    model = FacePCAModel(n_components=n_components)
    
    projected = model.train(data)
    
    assert projected.shape == (10, n_components)
    assert model.is_trained == True
    
    # Tester une nouvelle projection
    new_data = np.random.rand(1, 100)
    new_projected = model.project(new_data)
    assert new_projected.shape == (1, n_components)

def test_pca_untrained_error():
    model = FacePCAModel(n_components=5)
    with pytest.raises(ValueError):
        model.project(np.random.rand(1, 100))
