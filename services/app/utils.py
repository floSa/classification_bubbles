# Utilitaires du Dashboard Streamlit
"""
Connecteurs aux bases de données et fonctions de récupération des données.
"""

import os
import io
import logging
import sys
import pandas as pd
import psycopg2
import streamlit as st
from PIL import Image
from pymongo import MongoClient
from minio import Minio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import TimescaleConfig, MongoConfig, MinIOConfig

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# =============================================================================
# CONNEXIONS CACHÉES (Singleton Pattern via st.cache_resource)
# =============================================================================

@st.cache_resource
def get_pg_connection():
    """Connecteur persistant TimescaleDB."""
    try:
        conn = psycopg2.connect(**TimescaleConfig.get_dsn())
        return conn
    except Exception as e:
        st.sidebar.error(f"❌ TimescaleDB: {e}")
        return None


@st.cache_resource
def get_mongo_client():
    """Connecteur persistant MongoDB."""
    try:
        client = MongoClient(MongoConfig.get_uri(), serverSelectionTimeoutMS=2000)
        client.server_info()  # Test de connexion
        return client
    except Exception as e:
        st.sidebar.error(f"❌ MongoDB: {e}")
        return None


@st.cache_resource
def get_minio_client():
    """Connecteur persistant MinIO."""
    try:
        client = Minio(
            MinIOConfig.ENDPOINT,
            access_key=MinIOConfig.ACCESS_KEY,
            secret_key=MinIOConfig.SECRET_KEY,
            secure=MinIOConfig.SECURE
        )
        return client
    except Exception as e:
        st.sidebar.error(f"❌ MinIO: {e}")
        return None

# =============================================================================
# DATA FETCHERS
# =============================================================================

def fetch_audio_signal(conn, limit: int = 2000) -> pd.DataFrame:
    """Récupère les dernières données audio brutes."""
    if not conn:
        return pd.DataFrame()
    
    # Utilisation de paramètres pour éviter l'injection SQL
    query = """
        SELECT time, amplitude, label 
        FROM audio_data 
        ORDER BY time DESC 
        LIMIT %s;
    """
    try:
        df = pd.read_sql(query, conn, params=(limit,))
        return df
    except Exception:
        return pd.DataFrame()


def fetch_latest_events(mongo_client, limit: int = 4) -> list:
    """Récupère les derniers événements traités avec leurs prédictions."""
    if not mongo_client:
        return []
    
    try:
        db = mongo_client[MongoConfig.DATABASE]
        coll = db[MongoConfig.COLLECTION_BUBBLES]
        
        query = {"processed": True, "s3_key": {"$exists": True}}
        projection = {
            "timestamp": 1, 
            "label": 1, 
            "s3_key": 1, 
            "processed_at": 1,
            "prediction": 1  # Inclure les prédictions si disponibles
        }
        
        return list(coll.find(query, projection).sort("timestamp", -1).limit(limit))
    except Exception:
        return []


def fetch_spectrogram_image(minio_client, object_key: str):
    """Récupère et convertit une image depuis MinIO."""
    if not minio_client or not object_key:
        return None
    
    try:
        response = minio_client.get_object(MinIOConfig.BUCKET, object_key)
        img_data = response.read()
        response.close()
        response.release_conn()
        return Image.open(io.BytesIO(img_data))
    except Exception:
        return None


# =============================================================================
# TRAINING PROGRESS FETCHERS
# =============================================================================

def fetch_data_generation_stats(mongo_client) -> dict:
    """Récupère les statistiques de génération de données pour le training."""
    if not mongo_client:
        return {}
    
    try:
        db = mongo_client[MongoConfig.DATABASE]
        coll = db[MongoConfig.COLLECTION_BUBBLES]
        
        # Comptage total et par statut
        total = coll.count_documents({})
        processed = coll.count_documents({"processed": True})
        pending = coll.count_documents({"processed": False})
        errors = coll.count_documents({"processed": "error"})
        
        # Comptage par label (pour équilibre des classes)
        label_counts = {}
        for label in [0, 20, 40, 60, 80]:
            label_counts[label] = coll.count_documents({"label": label, "processed": True})
        
        return {
            "total": total,
            "processed": processed,
            "pending": pending,
            "errors": errors,
            "by_label": label_counts
        }
    except Exception as e:
        logging.error(f"Erreur fetch_data_generation_stats: {e}")
        return {}


def fetch_training_status() -> dict:
    """
    Vérifie l'état de l'entraînement en regardant le fichier modèle.
    Retourne le status et les infos du modèle si disponible.
    """
    import os
    model_path = "/app/models/bubble_classifier.pth"
    
    status = {
        "model_exists": False,
        "model_size": 0,
        "last_modified": None
    }
    
    try:
        if os.path.exists(model_path):
            stat = os.stat(model_path)
            status["model_exists"] = True
            status["model_size"] = stat.st_size / (1024 * 1024)  # MB
            from datetime import datetime
            status["last_modified"] = datetime.fromtimestamp(stat.st_mtime)
    except Exception:
        pass
    
    return status


# =============================================================================
# PHASE DETECTION & INFERENCE
# =============================================================================

def get_system_phase(mongo_client) -> str:
    """
    Détecte la phase actuelle du système.
    
    Returns:
        'generating' : Données en cours de génération
        'training' : Données prêtes, entraînement en cours
        'ready' : Modèle entraîné, prêt pour l'inférence
    """
    stats = fetch_data_generation_stats(mongo_client)
    training_status = fetch_training_status()
    
    # Si le modèle existe → ready
    if training_status.get("model_exists"):
        return "ready"
    
    # Si pas assez de données ou données en cours de génération
    total = stats.get("total", 0)
    processed = stats.get("processed", 0)
    pending = stats.get("pending", 0)
    
    # Seuil minimum pour lancer le training (au moins 50 échantillons traités)
    MIN_SAMPLES = 50
    
    if processed < MIN_SAMPLES or pending > 0:
        return "generating"
    
    # Données prêtes mais pas de modèle → training en cours
    return "training"


def fetch_latest_prediction(mongo_client) -> dict:
    """
    Récupère la dernière prédiction disponible.
    
    Returns:
        dict avec 'label', 'confidence', 'ground_truth', 's3_key', 'timestamp'
    """
    if not mongo_client:
        return {}
    
    try:
        db = mongo_client[MongoConfig.DATABASE]
        coll = db[MongoConfig.COLLECTION_BUBBLES]
        
        # Chercher le dernier document avec une prédiction
        query = {"prediction": {"$exists": True}}
        doc = coll.find_one(query, sort=[("timestamp", -1)])
        
        if doc and doc.get("prediction"):
            return {
                "label": doc["prediction"].get("label"),
                "confidence": doc["prediction"].get("confidence", 0),
                "ground_truth": doc.get("label"),
                "s3_key": doc.get("s3_key"),
                "timestamp": doc.get("timestamp")
            }
    except Exception as e:
        logging.error(f"Erreur fetch_latest_prediction: {e}")
    
    return {}


def fetch_latest_bubble_complete(mongo_client) -> dict:
    """
    Récupère le dernier document bulle complet avec raw_audio, fft_data et prediction.
    
    Returns:
        dict avec 'raw_audio', 'fft_data', 's3_key', 'prediction', 'label', 'timestamp'
    """
    if not mongo_client:
        return {}
    
    try:
        db = mongo_client[MongoConfig.DATABASE]
        coll = db[MongoConfig.COLLECTION_BUBBLES]
        
        # Chercher le dernier document traité
        query = {"processed": True, "s3_key": {"$exists": True}}
        doc = coll.find_one(query, sort=[("timestamp", -1)])
        
        if doc:
            return {
                "_id": str(doc["_id"]),
                "raw_audio": doc.get("raw_audio", []),
                "fft_data": doc.get("fft_data", {}),
                "s3_key": doc.get("s3_key"),
                "label": doc.get("label"),
                "prediction": doc.get("prediction"),
                "timestamp": doc.get("timestamp"),
                "sample_rate": doc.get("sample_rate", 4009)
            }
    except Exception as e:
        logging.error(f"Erreur fetch_latest_bubble_complete: {e}")
    
    return {}


def compute_fft(signal_data: pd.Series, sample_rate: int = 4009) -> tuple:
    """
    Calcule la FFT d'un signal audio.
    
    Args:
        signal_data: Série pandas avec les amplitudes
        sample_rate: Fréquence d'échantillonnage (après décimation)
    
    Returns:
        (frequencies, magnitudes) pour le tracé
    """
    import numpy as np
    
    if signal_data is None or len(signal_data) == 0:
        return [], []
    
    try:
        signal = signal_data.values
        n = len(signal)
        
        # FFT
        fft_result = np.fft.rfft(signal)
        magnitudes = np.abs(fft_result) / n
        frequencies = np.fft.rfftfreq(n, d=1/sample_rate)
        
        return frequencies, magnitudes
    except Exception:
        return [], []
