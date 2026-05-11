# Gestion centralisée des connexions aux bases de données
"""
Ce module fournit des fonctions pour obtenir des connexions robustes
vers TimescaleDB, MongoDB et MinIO avec gestion des retries.
"""

import time
import logging
import psycopg2
from pymongo import MongoClient
from minio import Minio

from .config import TimescaleConfig, MongoConfig, MinIOConfig

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# =============================================================================
# TIMESCALEDB (PostgreSQL)
# =============================================================================

def get_timescale_connection(max_retries: int = 10, retry_delay: float = 5.0):
    """
    Crée une connexion robuste à TimescaleDB avec retry automatique.
    
    Args:
        max_retries: Nombre maximum de tentatives
        retry_delay: Délai entre les tentatives (secondes)
    
    Returns:
        Connection psycopg2 ou lève une exception après échec
    """
    logger = logging.getLogger("TimescaleDB_Conn")
    
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(**TimescaleConfig.get_dsn())
            conn.autocommit = False
            logger.info("✅ Connexion TimescaleDB établie")
            return conn
        except Exception as e:
            logger.warning(f"⚠️ Tentative {attempt + 1}/{max_retries} échouée: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    raise ConnectionError("❌ Impossible de se connecter à TimescaleDB après plusieurs tentatives")

# =============================================================================
# MONGODB
# =============================================================================

def get_mongo_client(timeout_ms: int = 5000) -> MongoClient:
    """
    Crée un client MongoDB.
    
    Args:
        timeout_ms: Timeout de sélection du serveur en millisecondes
    
    Returns:
        Client MongoDB connecté
    """
    logger = logging.getLogger("MongoDB_Conn")
    
    try:
        client = MongoClient(
            MongoConfig.get_uri(), 
            serverSelectionTimeoutMS=timeout_ms
        )
        # Test de connexion
        client.server_info()
        logger.info("✅ Connexion MongoDB établie")
        return client
    except Exception as e:
        logger.error(f"❌ Erreur MongoDB: {e}")
        raise


def get_mongo_collection(collection_name: str = None):
    """
    Raccourci pour obtenir directement une collection MongoDB.
    
    Args:
        collection_name: Nom de la collection (défaut: bubbles)
    
    Returns:
        Collection MongoDB
    """
    client = get_mongo_client()
    db = client[MongoConfig.DATABASE]
    return db[collection_name or MongoConfig.COLLECTION_BUBBLES]

# =============================================================================
# MINIO (S3-Compatible)
# =============================================================================

def get_minio_client(ensure_bucket: bool = True) -> Minio:
    """
    Crée un client MinIO et s'assure que le bucket existe.
    
    Args:
        ensure_bucket: Si True, crée le bucket s'il n'existe pas
    
    Returns:
        Client MinIO configuré
    """
    logger = logging.getLogger("MinIO_Conn")
    
    client = Minio(
        MinIOConfig.ENDPOINT,
        access_key=MinIOConfig.ACCESS_KEY,
        secret_key=MinIOConfig.SECRET_KEY,
        secure=MinIOConfig.SECURE
    )
    
    if ensure_bucket:
        if not client.bucket_exists(MinIOConfig.BUCKET):
            client.make_bucket(MinIOConfig.BUCKET)
            logger.info(f"✅ Bucket '{MinIOConfig.BUCKET}' créé")
        else:
            logger.info(f"✅ Bucket '{MinIOConfig.BUCKET}' existe déjà")
    
    return client
