# Service d'Extraction - Détection et découpage des bulles
"""
Ce service surveille TimescaleDB pour détecter les événements acoustiques (bulles)
et les enregistre dans MongoDB pour traitement ultérieur.
"""

import time
import logging
import sys
import os
import numpy as np
import scipy.signal
from datetime import datetime, timedelta, timezone
from pymongo import ASCENDING

# Ajout du chemin parent pour importer common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import SAMPLE_RATE, DECIMATE_FACTOR, MongoConfig
from common.db_connections import get_timescale_connection, get_mongo_client

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Extraction_Service")

# =============================================================================
# CONFIGURATION DSP (Traitement du Signal)
# =============================================================================

BATCH_DURATION = 1.0       # On traite par blocs de 1 seconde pour fluidité temps réel
MARGIN_DURATION = 0.2       # Marge réduite pour fenêtres courtes (0.5 -> 0.2)
BUBBLE_LENGTH = 0.2         # Durée extraite par bulle (secondes)
PRE_ROLL = 0.05             # Temps gardé avant le pic (pour avoir l'attaque)
PEAK_HEIGHT = 0.3           # Seuil minimum d'amplitude pour considérer une bulle

# Fréquence effective après décimation
EFFECTIVE_FS = SAMPLE_RATE / DECIMATE_FACTOR
MIN_PEAK_DISTANCE = int(0.15 * EFFECTIVE_FS)  # Pas de bulles trop proches

# =============================================================================
# FONCTIONS MONGODB
# =============================================================================

def init_mongo_indexes(db):
    """Crée les index pour accélérer les recherches futures."""
    db[MongoConfig.COLLECTION_BUBBLES].create_index([("processed", ASCENDING)])
    db[MongoConfig.COLLECTION_BUBBLES].create_index([("timestamp", ASCENDING)])


def get_last_checkpoint(db, pg_conn):
    """Récupère la date de fin du dernier traitement."""
    state = db[MongoConfig.COLLECTION_STATE].find_one({"_id": "extraction_cursor"})
    
    if state:
        return state["last_processed_timestamp"].replace(tzinfo=timezone.utc)
    
    # Si aucun état, on ne cherche PAS dans tout l'historique (trop lent).
    # On reprend le flux temps réel avec une marge de sécurité de 30 secondes.
    logger.warning("⚠️ Aucun checkpoint trouvé. Démarrage au temps actuel - 30s.")
    return datetime.now(timezone.utc) - timedelta(seconds=30)


def update_checkpoint(db, new_timestamp):
    """Sauvegarde où on s'est arrêté."""
    db[MongoConfig.COLLECTION_STATE].update_one(
        {"_id": "extraction_cursor"},
        {"$set": {"last_processed_timestamp": new_timestamp}},
        upsert=True
    )

# =============================================================================
# FONCTIONS DE TRAITEMENT
# =============================================================================

def fetch_audio_segment(conn, start_time, duration_with_margin):
    """
    Récupère le signal audio brut depuis TimescaleDB.
    Retourne : (timestamps, amplitudes, labels) ou (None, None, None) si vide.
    """
    end_time = start_time + timedelta(seconds=duration_with_margin)
    
    query = """
        SELECT time, amplitude, label 
        FROM audio_data 
        WHERE time >= %s AND time < %s
        ORDER BY time ASC;
    """
    
    with conn.cursor() as cur:
        cur.execute(query, (start_time, end_time))
        rows = cur.fetchall()
        
    if not rows:
        return None, None, None

    # Conversion en Numpy pour traitement rapide
    times = np.array([r[0] for r in rows])
    amps = np.array([r[1] for r in rows], dtype=np.float32)
    labels = np.array([r[2] for r in rows], dtype=np.int16)
    
    return times, amps, labels


def process_and_extract(times, amps, labels):
    """
    Cœur du système : Détection de pics et découpage.
    
    Returns:
        Liste de documents MongoDB représentant les bulles extraites
    """
    extracted_bubbles = []
    
    # 1. Détection des pics (Bubbles)
    peaks, properties = scipy.signal.find_peaks(
        amps, 
        height=PEAK_HEIGHT, 
        distance=MIN_PEAK_DISTANCE
    )
    
    samples_len = int(BUBBLE_LENGTH * EFFECTIVE_FS)
    pre_roll_len = int(PRE_ROLL * EFFECTIVE_FS)
    
    for p_idx in peaks:
        # 2. Calcul des indices de découpe
        start_idx = p_idx - pre_roll_len
        end_idx = start_idx + samples_len
        
        # 3. Vérification des bords
        if start_idx < 0 or end_idx > len(amps):
            continue
            
        # 4. Extraction
        bubble_amp = amps[start_idx:end_idx]
        bubble_time = times[p_idx]
        bubble_label = labels[p_idx]
        
        # Récupération du peak height
        peak_mask = np.where(peaks == p_idx)[0]
        peak_height_val = properties["peak_heights"][peak_mask[0]] if len(peak_mask) > 0 else 0.0
        
        # Création du document Mongo
        doc = {
            "timestamp": bubble_time,
            "label": int(bubble_label),
            "amplitude_max": float(peak_height_val),
            "duration_sec": BUBBLE_LENGTH,
            "sample_rate": int(EFFECTIVE_FS),
            "raw_audio": bubble_amp.tolist(),
            "processed": False,
            "s3_spectrogram_path": None
        }
        extracted_bubbles.append(doc)
        
    return extracted_bubbles


def main():
    logger.info("🚀 Démarrage du service d'Extraction...")
    
    # Initialisation des connexions
    try:
        mongo_client = get_mongo_client()
        db = mongo_client[MongoConfig.DATABASE]
        init_mongo_indexes(db)
        
        pg_conn = get_timescale_connection()
    except Exception as e:
        logger.error(f"FATAL ERROR INIT: {e}")
        exit(1)
    
    try:
        consecutive_empty = 0  # Compteur de tentatives infructueuses consécutives
        MAX_EMPTY_RETRIES = 10  # Après 10 tentatives vides (~20s), on saute au présent
        
        while True:
            # 1. Où en sommes-nous ?
            current_cursor = get_last_checkpoint(db, pg_conn)
            
            target_next_cursor = current_cursor + timedelta(seconds=BATCH_DURATION)
            
            # Sécurité : on attend que Timescale ait fini d'écrire
            # On réduit la latence de sécurité à 0.5s pour le mode temps réel
            safety_now = datetime.now(timezone.utc) - timedelta(seconds=0.5)
            
            # 2. Logique Turbo vs Wait
            if target_next_cursor > safety_now:
                wait_seconds = (target_next_cursor - safety_now).total_seconds()
                if wait_seconds > 0:
                    time.sleep(1.0)
                    continue

            # 3. Récupération des données (Batch + Marge)
            times, amps, labels = fetch_audio_segment(
                pg_conn, 
                current_cursor, 
                BATCH_DURATION + MARGIN_DURATION
            )
            
            # Si pas de données - gérer le gap temporel
            if times is None or len(times) < 10:
                consecutive_empty += 1
                logger.info(f"⏳ Pas de données à {current_cursor}. Tentative {consecutive_empty}/{MAX_EMPTY_RETRIES}...")
                
                # Après trop de tentatives, sauter au temps actuel (gap temporel détecté)
                if consecutive_empty >= MAX_EMPTY_RETRIES:
                    new_cursor = datetime.now(timezone.utc) - timedelta(seconds=5)
                    logger.warning(f"⚠️ Gap temporel détecté ! Saut de {current_cursor} à {new_cursor}")
                    update_checkpoint(db, new_cursor)
                    consecutive_empty = 0
                else:
                    time.sleep(2.0)
                continue
            
            # Vérification de complétude
            last_point_time = times[-1].replace(tzinfo=timezone.utc)
            if last_point_time < target_next_cursor:
                # DONNÉES INCOMPLÈTES
                consecutive_empty += 1
                logger.info(f"⚠️ Données partielles à {current_cursor} (Fin: {last_point_time.time()} < Cible: {target_next_cursor.time()}). Tentative {consecutive_empty}/{MAX_EMPTY_RETRIES}...")
                
                if consecutive_empty >= MAX_EMPTY_RETRIES:
                    new_cursor = datetime.now(timezone.utc) - timedelta(seconds=5)
                    logger.warning(f"⚠️ Gap temporel détecté (sur incomplet) ! Saut de {current_cursor} à {new_cursor}")
                    update_checkpoint(db, new_cursor)
                    consecutive_empty = 0
                else:
                    time.sleep(0.5)
                continue

            # Reset du compteur si on a des données COMPLÈTES
            consecutive_empty = 0

            # 4. Traitement
            bubbles = process_and_extract(times, amps, labels)
            
            # 5. Sauvegarde Mongo
            if bubbles:
                db[MongoConfig.COLLECTION_BUBBLES].insert_many(bubbles)
                logger.info(f"✅ Extrait {len(bubbles)} bulles entre {current_cursor.time()} et {target_next_cursor.time()}")
            else:
                logger.info(f"∅ Aucune bulle détectée sur ce segment.")

            # 6. Mise à jour du Checkpoint
            update_checkpoint(db, target_next_cursor)

    except KeyboardInterrupt:
        logger.info("Arrêt du service.")
    finally:
        pg_conn.close()


if __name__ == "__main__":
    main()