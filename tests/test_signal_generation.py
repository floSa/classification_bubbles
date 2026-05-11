# Tests de génération de signal audio
"""
Tests unitaires pour le module signal_processing.
"""

import sys
import os
import numpy as np

# Ajout du chemin pour importer les modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'services'))

from common.signal_processing import generate_signal
from common.config import SAMPLE_RATE, CLOGGING_LEVELS


class TestSignalGeneration:
    """Tests pour la fonction generate_signal."""
    
    def test_generate_signal_returns_correct_shape(self):
        """Vérifie que le signal généré a la bonne forme."""
        duration = 1.0
        t, signal = generate_signal(0, duration)
        
        expected_samples = int(SAMPLE_RATE * duration)
        assert len(t) == expected_samples, f"Time axis devrait avoir {expected_samples} samples"
        assert len(signal) == expected_samples, f"Signal devrait avoir {expected_samples} samples"
    
    def test_generate_signal_returns_float16(self):
        """Vérifie que le signal est en float16 pour économiser la mémoire."""
        _, signal = generate_signal(0, 0.5)
        assert signal.dtype == np.float16, "Signal devrait être en float16"
    
    def test_generate_signal_within_bounds(self):
        """Vérifie que le signal reste dans les limites [-1, 1]."""
        for level in CLOGGING_LEVELS:
            _, signal = generate_signal(level, 0.5)
            assert signal.min() >= -1.0, f"Signal niveau {level} ne devrait pas descendre sous -1"
            assert signal.max() <= 1.0, f"Signal niveau {level} ne devrait pas dépasser 1"
    
    def test_generate_signal_different_levels_have_different_characteristics(self):
        """Vérifie que les différents niveaux produisent des signaux différents."""
        signals = {}
        for level in CLOGGING_LEVELS:
            _, signal = generate_signal(level, 1.0)
            signals[level] = np.std(signal.astype(np.float32))  # Convert to float32 for std
        
        # Le bruit devrait augmenter avec le niveau de bouchage
        # (pas toujours strict à cause du caractère aléatoire, mais tendance générale)
        assert signals[0] < signals[80], "Niveau 0 devrait avoir moins de variabilité que niveau 80"
    
    def test_generate_signal_time_axis_correct(self):
        """Vérifie que l'axe temporel est correct."""
        duration = 2.0
        t, _ = generate_signal(40, duration)
        
        assert t[0] == 0.0, "Le temps devrait commencer à 0"
        assert t[-1] < duration, "Le temps ne devrait pas atteindre la durée (endpoint=False)"
        
        # Vérifier l'espacement
        dt = t[1] - t[0]
        expected_dt = 1.0 / SAMPLE_RATE
        assert abs(dt - expected_dt) < 1e-10, f"Espacement temporel incorrect: {dt} vs {expected_dt}"


def test_signal_generation_basic():
    """Test simple pour pytest discovery."""
    t, signal = generate_signal(0, 0.1)
    assert len(signal) > 0
    assert signal.dtype == np.float16


def test_all_clogging_levels():
    """Test que tous les niveaux de bouchage fonctionnent."""
    for level in CLOGGING_LEVELS:
        t, signal = generate_signal(level, 0.1)
        assert len(signal) > 0, f"Niveau {level} devrait produire un signal"
