# Configuration centralisée pour tous les services
"""
Ce module centralise toutes les constantes et configurations partagées.
Les variables d'environnement sont chargées une seule fois ici.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONSTANTES DE TRAITEMENT DU SIGNAL
# =============================================================================

# Fréquence d'échantillonnage native (qualité CD)
SAMPLE_RATE = 44100

# Durée d'un chunk audio en secondes
CHUNK_DURATION = 1.0

# Facteur de décimation pour réduire le volume de données
# 44100 / 11 ≈ 4009 Hz (suffisant pour Nyquist à 2000Hz, bulles max 1200Hz)
DECIMATE_FACTOR = 11

# Échelle d'amplitude pour normalisation
AMPLITUDE_SCALE = 0.8

# Taille standard des images pour le CNN (MobileNet/ResNet)
IMG_SIZE = (224, 224)

# =============================================================================
# PARAMÈTRES PHYSIQUES DES BULLES (Modèle de simulation)
# =============================================================================
#
# Chaque classe est une DISTRIBUTION (moyennes + écarts-types) et non un point
# fixe. Les distributions de classes adjacentes se chevauchent volontairement
# pour forcer le classifieur à apprendre une représentation conjointe plutôt
# qu'à exploiter une feature unique trivialement séparable.
#
# Champs:
#   freq_mean / freq_std        : fréquence fondamentale d'une bulle (Hz, gauss.)
#   interval_mean / interval_std: intervalle entre deux bulles (s, gauss.)
#   decay_mean / decay_std      : taux de décroissance de l'enveloppe exp (1/s)
#   harmonics_range             : (min, max) inclus, nombre d'harmoniques ajoutés
#   noise_std                   : écart-type du bruit de fond coloré (pink)
#   amp_jitter                  : variation d'amplitude par bulle (± fraction)
#   double_prob                 : probabilité d'un "double burst" (cavitation jumelée)

BUBBLE_PARAMS = {
    0:  {"freq_mean": 850,  "freq_std": 130,
         "interval_mean": 0.60, "interval_std": 0.18,
         "decay_mean": 22.0, "decay_std": 4.0,
         "harmonics_range": (0, 1),
         "noise_std": 0.045, "amp_jitter": 0.30,
         "double_prob": 0.05},
    20: {"freq_mean": 900,  "freq_std": 150,
         "interval_mean": 0.50, "interval_std": 0.17,
         "decay_mean": 20.0, "decay_std": 5.0,
         "harmonics_range": (0, 2),
         "noise_std": 0.052, "amp_jitter": 0.35,
         "double_prob": 0.07},
    40: {"freq_mean": 970,  "freq_std": 170,
         "interval_mean": 0.42, "interval_std": 0.15,
         "decay_mean": 18.0, "decay_std": 5.0,
         "harmonics_range": (1, 2),
         "noise_std": 0.060, "amp_jitter": 0.40,
         "double_prob": 0.10},
    60: {"freq_mean": 1040, "freq_std": 180,
         "interval_mean": 0.34, "interval_std": 0.12,
         "decay_mean": 16.0, "decay_std": 6.0,
         "harmonics_range": (1, 3),
         "noise_std": 0.068, "amp_jitter": 0.45,
         "double_prob": 0.13},
    80: {"freq_mean": 1110, "freq_std": 200,
         "interval_mean": 0.28, "interval_std": 0.10,
         "decay_mean": 14.0, "decay_std": 6.0,
         "harmonics_range": (1, 3),
         "noise_std": 0.075, "amp_jitter": 0.50,
         "double_prob": 0.18},
}

# Labels disponibles (niveaux de bouchage en %)
CLOGGING_LEVELS = [0, 20, 40, 60, 80]

# Taux de bruit de label (0..1) : fraction de chunks dont le label stocké
# diffère de la classe réelle utilisée pour la génération. Utile pour stresser
# le modèle et éviter une val_acc artificielle à 100%.
LABEL_NOISE_RATE = float(os.getenv("LABEL_NOISE_RATE", "0.0"))

# Mapping label -> index pour le modèle ML
LABEL_TO_INDEX = {0: 0, 20: 1, 40: 2, 60: 3, 80: 4}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}

# =============================================================================
# CONFIGURATION TIMESCALEDB
# =============================================================================

class TimescaleConfig:
    """Configuration pour la connexion TimescaleDB."""
    HOST = os.getenv("TIMESCALE_HOST", "localhost")
    PORT = os.getenv("TIMESCALE_PORT", "5432")
    USER = os.getenv("TIMESCALE_USER", "postgres")
    PASSWORD = os.getenv("TIMESCALE_PASSWORD", "password")
    DATABASE = os.getenv("TIMESCALE_DB", "bubble_db")
    
    @classmethod
    def get_dsn(cls) -> dict:
        """Retourne le dictionnaire de connexion pour psycopg2."""
        return {
            "host": cls.HOST,
            "port": cls.PORT,
            "user": cls.USER,
            "password": cls.PASSWORD,
            "database": cls.DATABASE
        }

# =============================================================================
# CONFIGURATION MONGODB
# =============================================================================

class MongoConfig:
    """Configuration pour la connexion MongoDB."""
    HOST = os.getenv("MONGO_HOST", "localhost")
    PORT = os.getenv("MONGO_PORT", "27017")
    USER = os.getenv("MONGO_USER", "root")
    PASSWORD = os.getenv("MONGO_PASSWORD", "password")
    DATABASE = "bubble_project"
    COLLECTION_BUBBLES = "bubbles"
    COLLECTION_STATE = "state_checkpoints"
    
    @classmethod
    def get_uri(cls) -> str:
        """Retourne l'URI de connexion MongoDB."""
        return f"mongodb://{cls.USER}:{cls.PASSWORD}@{cls.HOST}:{cls.PORT}/"

# =============================================================================
# CONFIGURATION MINIO (S3-Compatible)
# =============================================================================

class MinIOConfig:
    """Configuration pour la connexion MinIO."""
    ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    BUCKET = "spectrograms"
    SECURE = False  # True si HTTPS
