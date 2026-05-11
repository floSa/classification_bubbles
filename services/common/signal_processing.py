# Fonctions de traitement du signal audio
"""
Ce module contient les fonctions de génération et manipulation
des signaux audio simulés représentant les bulles.
"""

import random
import logging
import numpy as np
from psycopg2.extras import execute_values

from .config import (
    SAMPLE_RATE, 
    AMPLITUDE_SCALE, 
    DECIMATE_FACTOR,
    BUBBLE_PARAMS
)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Signal_Processing")

# =============================================================================
# GÉNÉRATION DE SIGNAL
# =============================================================================

def generate_signal(clogging_level: int, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Génère un signal audio simulant des bulles pour un niveau de bouchage donné.
    
    Le modèle physique simule :
    - Bruit de fond gaussien (turbulence)
    - Impulsions de bulles avec enveloppe exponentielle décroissante
    - Fréquence et chaos variant selon le niveau de bouchage
    
    Args:
        clogging_level: Niveau de bouchage (0, 20, 40, 60, 80)
        duration: Durée du signal en secondes
    
    Returns:
        Tuple (time_axis, signal_float16)
    """
    # Axe temporel
    num_samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # 1. Bruit de fond (turbulence légère, proportionnelle au bouchage)
    # Réduit pour garder le signal visible même à 80%
    noise_std = 0.01 + (clogging_level / 800.0)  # Max ~0.11 au lieu de 0.42
    signal = np.random.normal(0, noise_std, size=t.shape)

    # 2. Paramètres physiques selon le niveau de bouchage
    base_freq, burst_interval, chaos = BUBBLE_PARAMS.get(clogging_level, (800, 0.8, 0.0))

    # 3. Injection de bulles (impulsions aléatoires)
    cursor = 0.0
    burst_duration = 0.1  # Durée d'une bulle en secondes
    burst_samples = int(burst_duration * SAMPLE_RATE)
    
    while cursor < duration:
        # Variation chaotique de l'intervalle (réduite pour garder la lisibilité)
        variation = random.uniform(-chaos * 0.3, chaos * 0.3) * burst_interval
        wait = max(0.15, burst_interval + variation)
        cursor += wait
        
        if cursor >= duration:
            break
        
        idx = int(cursor * SAMPLE_RATE)
        if idx + burst_samples >= len(t):
            continue

        # Création de la bulle
        local_t = np.linspace(0, burst_duration, burst_samples)
        freq = base_freq + random.uniform(-50, 50)  # Variation de fréquence
        envelope = np.exp(-local_t * 20)  # Décroissance exponentielle
        burst = np.sin(2 * np.pi * freq * local_t) * envelope * AMPLITUDE_SCALE
        
        # Ajout au signal
        signal[idx:idx + burst_samples] += burst

    # 4. Clipping et conversion en float16
    # À 80%, le signal peut saturer (simulation de cavitation)
    limit = 1.5 if clogging_level >= 80 else 1.0
    signal = np.clip(signal, -limit, limit)
    
    # Hard clip final pour rester dans [-1, 1]
    signal = np.clip(signal, -1.0, 1.0)
    
    return t, signal.astype(np.float16)

# =============================================================================
# FONCTIONS DE BASE DE DONNÉES
# =============================================================================

def is_db_populated(conn) -> bool:
    """
    Vérifie si la table audio_data contient déjà des données.
    
    Args:
        conn: Connexion psycopg2 active
    
    Returns:
        True si des données existent, False sinon
    """
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM audio_data LIMIT 1;")
        return cur.fetchone() is not None


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
