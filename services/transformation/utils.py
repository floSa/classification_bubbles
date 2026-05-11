# Utilitaires du Service Transformation
"""
Fonctions de création de spectrogrammes et gestion du stockage MinIO.
"""

import os
import io
import time
import logging
import numpy as np
import scipy.signal
from PIL import Image
from functools import wraps

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import IMG_SIZE, MinIOConfig
from common.db_connections import get_mongo_collection, get_minio_client

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Transformation_Utils")

# =============================================================================
# RETRY DECORATOR (Pour gestion backpressure MinIO)
# =============================================================================

def retry_with_backoff(max_attempts: int = 3, initial_delay: float = 1.0, max_delay: float = 10.0):
    """
    Décorateur pour retry avec backoff exponentiel.
    
    Args:
        max_attempts: Nombre maximum de tentatives
        initial_delay: Délai initial entre les tentatives
        max_delay: Délai maximum entre les tentatives
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"⚠️ Tentative {attempt + 1}/{max_attempts} échouée: {e}. Retry dans {delay:.1f}s...")
                        time.sleep(delay)
                        delay = min(delay * 2, max_delay)  # Backoff exponentiel
                    else:
                        logger.error(f"❌ Toutes les tentatives échouées pour {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator

# =============================================================================
# GÉNÉRATION DE SPECTROGRAMME
# =============================================================================

def create_spectrogram_image(raw_audio: list, fs: int) -> tuple[io.BytesIO, float, float]:
    """
    Convertit le signal brut en image spectrogramme optimisée pour le ML.
    
    Args:
        raw_audio: Liste des amplitudes audio
        fs: Fréquence d'échantillonnage
    
    Returns:
        Tuple (BytesIO, min_db, max_db)
    """
    signal = np.array(raw_audio)
    
    # 1. Calcul du Spectrogramme (STFT)
    f, t, Sxx = scipy.signal.spectrogram(signal, fs, nperseg=256, noverlap=128)
    
    # 2. Conversion en décibels (Log scale)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # 3. Normalisation Min-Max pour tenir dans 0-255
    s_min, s_max = Sxx_db.min(), Sxx_db.max()
    if s_max - s_min > 0:
        normalized = (Sxx_db - s_min) / (s_max - s_min) * 255
    else:
        normalized = np.zeros_like(Sxx_db)
    normalized = normalized.astype(np.uint8)
    
    # 4. Inversion de l'axe Y (basses fréquences en bas)
    normalized = np.flipud(normalized)
    
    # 5. Création de l'image avec Pillow
    image = Image.fromarray(normalized, mode='L')
    
    # 6. Redimensionnement à la taille standard CNN
    image = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    
    # 7. Sauvegarde en mémoire (Buffer)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr, s_min, s_max


def compute_fft_for_storage(raw_audio: list, fs: int, max_freq: int = 2000) -> dict:
    """
    Calcule la FFT du signal pour stockage dans MongoDB.
    
    Args:
        raw_audio: Liste des amplitudes audio
        fs: Fréquence d'échantillonnage
        max_freq: Fréquence maximale à conserver (Hz)
    
    Returns:
        dict avec 'frequencies' et 'magnitudes' (listes pour MongoDB)
    """
    signal = np.array(raw_audio)
    n = len(signal)
    
    # Calcul FFT
    fft_result = np.fft.rfft(signal)
    magnitudes = np.abs(fft_result) / n
    frequencies = np.fft.rfftfreq(n, d=1/fs)
    
    # Filtrer jusqu'à max_freq Hz
    mask = frequencies <= max_freq
    
    return {
        "frequencies": frequencies[mask].tolist(),
        "magnitudes": magnitudes[mask].tolist(),
        "max_freq": max_freq
    }

# =============================================================================
# UPLOAD MINIO AVEC BACKPRESSURE
# =============================================================================

@retry_with_backoff(max_attempts=3, initial_delay=1.0, max_delay=10.0)
def upload_to_minio(client, bucket: str, object_name: str, data: io.BytesIO, length: int):
    """
    Upload un fichier vers MinIO avec gestion des retries.
    
    Args:
        client: Client MinIO
        bucket: Nom du bucket
        object_name: Chemin de l'objet
        data: BytesIO contenant les données
        length: Taille des données
    """
    data.seek(0)  # Rembobiner au cas où
    client.put_object(
        bucket,
        object_name,
        data,
        length=length,
        content_type="image/png"
    )
    logger.debug(f"✅ Upload réussi: {object_name}")
