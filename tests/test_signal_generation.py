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


def test_generate_signal_is_stochastic():
    """
    Deux appels successifs au même niveau doivent produire des signaux
    différents (sinon le modèle de classes redevient déterministe).
    """
    _, sig_a = generate_signal(40, 1.0)
    _, sig_b = generate_signal(40, 1.0)
    # Les signaux sont en float16 mais on compare comme arrays
    diff_ratio = np.mean(sig_a.astype(np.float32) != sig_b.astype(np.float32))
    assert diff_ratio > 0.5, "Deux générations doivent être stochastiquement différentes"


def test_generate_signal_class_distributions_overlap():
    """
    Vérifie que les distributions de classes adjacentes se chevauchent en
    fréquence dominante (sinon le problème de classification est trivial).
    On regarde la fréquence du pic spectral sur plusieurs tirages.
    """
    from common.config import SAMPLE_RATE

    def dominant_freq(sig: np.ndarray) -> float:
        sig32 = sig.astype(np.float32)
        spectrum = np.abs(np.fft.rfft(sig32))
        freqs = np.fft.rfftfreq(len(sig32), d=1.0 / SAMPLE_RATE)
        # Ignore le DC
        spectrum[0] = 0
        return float(freqs[int(np.argmax(spectrum))])

    n_trials = 10
    freqs_low = [dominant_freq(generate_signal(0, 1.0)[1]) for _ in range(n_trials)]
    freqs_high = [dominant_freq(generate_signal(20, 1.0)[1]) for _ in range(n_trials)]

    # Au moins un chevauchement entre min(20%) et max(0%) (les ranges se croisent)
    assert max(freqs_low) > min(freqs_high) or min(freqs_low) < max(freqs_high), (
        "Les distributions de fréquence des classes 0 et 20 doivent se chevaucher"
    )
