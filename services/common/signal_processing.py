# Fonctions de traitement du signal audio
"""
Génération et insertion des signaux audio simulés représentant les bulles.

Modèle physique :
- Bruit de fond rose (1/f) pour simuler la turbulence d'écoulement.
- Bursts de bulles paramétrés par classe via des DISTRIBUTIONS stochastiques
  (et non des constantes) : fréquence, intervalle, décroissance et amplitude
  sont tirés à chaque bulle.
- Les distributions des 5 classes se chevauchent fortement : aucune feature
  scalaire ne permet à elle seule la séparation, le modèle doit apprendre une
  représentation conjointe.
- Harmoniques aléatoires (1 à 4 modes) pour s'éloigner d'une sinusoïde pure.
- Possibilité de double-burst (cavitation jumelée).
"""

import logging
import numpy as np
from psycopg2.extras import execute_values

from .config import (
    SAMPLE_RATE,
    AMPLITUDE_SCALE,
    BUBBLE_PARAMS,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Signal_Processing")


# =============================================================================
# BRUIT COLORÉ (1/f^beta)
# =============================================================================

def _colored_noise(n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """
    Génère un bruit coloré normalisé à std=1.

    beta=0 -> blanc, beta=1 -> rose (typique turbulence), beta=2 -> brownien.
    """
    freqs = np.fft.rfftfreq(n)
    spectrum = np.zeros_like(freqs)
    nz = freqs > 0
    spectrum[nz] = freqs[nz] ** (-beta / 2.0)
    spectrum[0] = 0.0

    phases = rng.uniform(0.0, 2.0 * np.pi, size=freqs.shape)
    complex_spectrum = spectrum * np.exp(1j * phases)

    noise = np.fft.irfft(complex_spectrum, n=n)
    s = float(noise.std())
    if s > 0:
        noise /= s
    return noise


# =============================================================================
# GÉNÉRATION DE SIGNAL
# =============================================================================

def generate_signal(clogging_level: int, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Génère un signal audio simulant des bulles pour un niveau de bouchage donné.

    Args:
        clogging_level: Niveau de bouchage (0, 20, 40, 60, 80)
        duration: Durée du signal en secondes

    Returns:
        Tuple (time_axis, signal_float16) — signal borné dans [-1, 1].
    """
    rng = np.random.default_rng()

    num_samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    p = BUBBLE_PARAMS.get(clogging_level)
    if p is None:
        # Fallback : prendre la classe la plus proche
        nearest = min(BUBBLE_PARAMS.keys(), key=lambda k: abs(k - clogging_level))
        p = BUBBLE_PARAMS[nearest]

    # 1) Bruit de fond rose (modèle de turbulence)
    signal = _colored_noise(num_samples, beta=1.0, rng=rng) * p["noise_std"]

    # 2) Séquence stochastique de bursts
    h_low, h_high = p["harmonics_range"]
    cursor = 0.0

    while cursor < duration:
        # Intervalle gaussien, borné inférieurement pour éviter une explosion de bulles
        wait = float(rng.normal(p["interval_mean"], p["interval_std"]))
        wait = max(0.05, wait)
        cursor += wait
        if cursor >= duration:
            break

        # Probabilité d'un double-burst (cavitation jumelée)
        offsets = [0.0]
        if rng.random() < p["double_prob"]:
            offsets.append(float(rng.uniform(0.02, 0.06)))

        for offset in offsets:
            burst_start = cursor + offset
            if burst_start >= duration:
                continue

            idx = int(burst_start * SAMPLE_RATE)

            # Paramètres stochastiques du burst
            base_freq = float(rng.normal(p["freq_mean"], p["freq_std"]))
            base_freq = float(np.clip(base_freq, 300.0, 2000.0))

            decay = float(rng.normal(p["decay_mean"], p["decay_std"]))
            decay = max(5.0, decay)

            amp = AMPLITUDE_SCALE * (1.0 + float(rng.uniform(-p["amp_jitter"], p["amp_jitter"])))

            # Durée adaptée à la décroissance (au moins ~4 constantes de temps)
            burst_duration = min(0.18, 4.5 / decay)
            burst_samples = int(burst_duration * SAMPLE_RATE)
            if burst_samples < 8 or idx + burst_samples >= num_samples:
                continue

            local_t = np.arange(burst_samples) / SAMPLE_RATE
            envelope = np.exp(-local_t * decay)

            # Fondamentale
            phase0 = float(rng.uniform(0.0, 2.0 * np.pi))
            burst = np.sin(2 * np.pi * base_freq * local_t + phase0)

            # Harmoniques (chacune avec léger détuning, amplitude et phase aléatoires)
            if h_high >= h_low:
                n_harmonics = int(rng.integers(h_low, h_high + 1))
            else:
                n_harmonics = 0

            for h in range(2, 2 + n_harmonics):
                h_freq = base_freq * h * float(rng.uniform(0.95, 1.05))
                if h_freq > SAMPLE_RATE * 0.45:  # éviter l'aliasing proche de Nyquist
                    continue
                h_amp = float(rng.uniform(0.20, 0.55))
                h_phase = float(rng.uniform(0.0, 2.0 * np.pi))
                burst += h_amp * np.sin(2 * np.pi * h_freq * local_t + h_phase)

            burst *= envelope * amp
            signal[idx:idx + burst_samples] += burst

    # 3) Clipping : à 80% on tolère une légère saturation (cavitation)
    limit = 1.5 if clogging_level >= 80 else 1.0
    signal = np.clip(signal, -limit, limit)
    signal = np.clip(signal, -1.0, 1.0)

    return t, signal.astype(np.float16)


# =============================================================================
# FONCTIONS DE BASE DE DONNÉES
# =============================================================================

def insert_batch(conn, buffer_data: list) -> bool:
    """
    Insère un lot de données audio dans TimescaleDB.

    Args:
        conn: Connexion psycopg2 active
        buffer_data: Liste de tuples (timestamp, amplitude, label)

    Returns:
        True si succès, False sinon
    """
    if not buffer_data:
        return True

    query = "INSERT INTO audio_data (time, amplitude, label) VALUES %s"

    try:
        with conn.cursor() as cur:
            execute_values(cur, query, buffer_data)
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Erreur d'insertion: {e}")
        return False
