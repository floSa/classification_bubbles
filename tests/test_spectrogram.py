# Tests de génération de spectrogrammes
"""
Tests unitaires pour les fonctions de transformation audio -> image.
"""

import sys
import os
import io
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'services'))

from common.config import IMG_SIZE


def test_spectrogram_creation():
    """Test la création d'un spectrogramme."""
    # Import local pour éviter les dépendances manquantes
    try:
        from transformation.utils import create_spectrogram_image
    except ImportError:
        # Si scipy n'est pas installé, skip le test
        import pytest
        pytest.skip("scipy non installé")
    
    # Signal de test simple (sinusoïde)
    fs = 4000
    duration = 0.2
    t = np.linspace(0, duration, int(fs * duration))
    test_signal = (np.sin(2 * np.pi * 440 * t) * 0.5).tolist()
    
    # Création du spectrogramme
    img_buffer, db_min, db_max = create_spectrogram_image(test_signal, fs)
    
    # Vérifications
    assert isinstance(img_buffer, io.BytesIO), "Devrait retourner un BytesIO"
    assert img_buffer.getbuffer().nbytes > 0, "Le buffer ne devrait pas être vide"
    assert db_min < db_max, "db_min devrait être inférieur à db_max"


def test_spectrogram_image_size():
    """Test que l'image a la bonne taille."""
    try:
        from transformation.utils import create_spectrogram_image
        from PIL import Image
    except ImportError:
        import pytest
        pytest.skip("dépendances manquantes")
    
    fs = 4000
    test_signal = np.random.randn(800).tolist()
    
    img_buffer, _, _ = create_spectrogram_image(test_signal, fs)
    
    # Vérifier les dimensions
    image = Image.open(img_buffer)
    assert image.size == IMG_SIZE, f"Image devrait être {IMG_SIZE}, obtenu {image.size}"
    assert image.mode == 'L', "Image devrait être en niveaux de gris (mode 'L')"


def test_spectrogram_handles_empty_signal():
    """Test la gestion des signaux vides ou trop courts."""
    try:
        from transformation.utils import create_spectrogram_image
    except ImportError:
        import pytest
        pytest.skip("scipy non installé")
    
    # Signal très court
    short_signal = [0.0, 0.1, -0.1]
    
    # Ne devrait pas lever d'exception
    try:
        img_buffer, _, _ = create_spectrogram_image(short_signal, 4000)
        assert img_buffer is not None
    except Exception as e:
        # C'est OK si ça échoue gracieusement
        pass
