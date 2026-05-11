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

# Mapping niveau de bouchage -> (fréquence_base, intervalle_bulles, indice_chaos)
BUBBLE_PARAMS = {
    0:  (800, 0.8, 0.02),   # Normal : bulles lentes, basse fréquence, très stable
    20: (850, 0.70, 0.05),  # Léger : légère augmentation
    40: (920, 0.55, 0.10),  # Modéré : bulles plus fréquentes
    60: (1050, 0.40, 0.15), # Sévère : haute fréquence, chaos modéré
    80: (1200, 0.25, 0.20)  # Critique : très haute fréquence, chaos contrôlé
}

# Labels disponibles (niveaux de bouchage en %)
CLOGGING_LEVELS = [0, 20, 40, 60, 80]

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
