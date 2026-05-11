# Script de population initiale (Historique)
"""
Génère l'historique des données en mode Turbo.
NETTOIE l'environnement au démarrage.
"""
import sys
import os
import time
import json
import logging
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import CHUNK_DURATION, DECIMATE_FACTOR, SAMPLE_RATE, CLOGGING_LEVELS
from common.db_connections import get_timescale_connection
from common.signal_processing import generate_signal, insert_batch

# Logger spécifique
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Initial_Pop")

# Chemins
PROGRESS_DIR = "/app/models"
ACQ_FILE = os.path.join(PROGRESS_DIR, "acquisition_progress.json")
TRAINING_FILE = os.path.join(PROGRESS_DIR, "training_progress.json")
MODEL_FILE = os.path.join(PROGRESS_DIR, "bubble_classifier.pth")

def write_progress(current, total, status):
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    data = {
        "current": current, "total": total,
        "percentage": round((current/total)*100, 1) if total else 0,
        "status": status
    }
    with open(ACQ_FILE, "w") as f:
        json.dump(data, f)


def reset_all_progress_files():
    """
    Réinitialise TOUS les fichiers de progression au démarrage d'une nouvelle génération.
    Cela garantit que le Dashboard affiche un état cohérent.
    """
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    
    # Reset acquisition progress
    acq_data = {
        "current": 0, "total": 0,
        "percentage": 0.0,
        "status": "starting"
    }
    with open(ACQ_FILE, "w") as f:
        json.dump(acq_data, f)
    logger.info(f"🔄 Fichier {ACQ_FILE} réinitialisé")
    
    # Reset training progress (le training n'a pas encore commencé)
    training_data = {
        "current_epoch": 0,
        "total_epochs": 10,
        "status": "waiting_data",
        "message": "En attente de la génération de données...",
        "loss": 0.0
    }
    with open(TRAINING_FILE, "w") as f:
        json.dump(training_data, f)
    logger.info(f"🔄 Fichier {TRAINING_FILE} réinitialisé")

def clean_environment(conn):
    """
    Nettoie DB et fichiers.
    ATTENTION: Cette fonction est DÉSACTIVÉE par défaut pour préserver les données.
    Utiliser FORCE_CLEAN=true pour forcer le nettoyage.
    """
    force_clean = os.environ.get("FORCE_CLEAN", "false").lower() == "true"
    
    if not force_clean:
        logger.info("ℹ️ Nettoyage désactivé (FORCE_CLEAN != true)")
        return
    
    logger.info("🧹 Nettoyage FORCÉ de l'environnement...")
    
    # 1. Fichiers (on ne supprime que le modèle si explicitement demandé)
    for f in [MODEL_FILE]:
        if os.path.exists(f): 
            try: 
                os.remove(f)
                logger.info(f"🗑️ Fichier supprimé: {f}")
            except Exception as e:
                logger.warning(f"Erreur suppression {f}: {e}")
    
    # 2. Base de données
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE audio_data;")
        conn.commit()
        logger.info("✅ DB Vidée.")
    except Exception as e:
        logger.warning(f"Erreur trunk DB: {e}")

def run():
    conn = get_timescale_connection()
    
    # IMPORTANT: Réinitialiser les fichiers de progression AVANT tout
    # Cela garantit que le Dashboard affiche l'état correct (génération en cours)
    reset_all_progress_files()
    
    # Nettoyage optionnel (désactivé par défaut)
    clean_environment(conn)
    
    logger.info("🚀 Démarrage Génération Historique (1h)...")
    SIMULATION_HOURS = 1
    total_iterations = int(SIMULATION_HOURS * 3600 / CHUNK_DURATION)
    
    virtual_clock = datetime.now(timezone.utc) - timedelta(hours=SIMULATION_HOURS)
    
    level_idx = 0
    buffer = []
    BATCH = 50000 
    
    # Init progress
    write_progress(0, total_iterations, "generating")
    
    current_iter = 0
    counter = 0
    samples_per_level = 200
    
    try:
        while current_iter < total_iterations:
            if counter >= samples_per_level:
                level_idx = (level_idx + 1) % len(CLOGGING_LEVELS)
                counter = 0
            
            level = CLOGGING_LEVELS[level_idx]
            
            # Génération
            _, signal = generate_signal(level, CHUNK_DURATION)
            
            # Préparation
            reduced_sig = signal[::DECIMATE_FACTOR]
            time_step = 1.0 / (SAMPLE_RATE / DECIMATE_FACTOR)
            base_ts = virtual_clock.timestamp()
            
            for i, val in enumerate(reduced_sig):
                ts = datetime.fromtimestamp(base_ts + i * time_step, timezone.utc)
                buffer.append((ts, float(val), int(level)))
            
            if len(buffer) >= BATCH:
                insert_batch(conn, buffer)
                buffer = []
                write_progress(current_iter, total_iterations, "generating")
                logger.info(f"📊 {current_iter}/{total_iterations} chunks")
            
            virtual_clock += timedelta(seconds=CHUNK_DURATION)
            counter += 1
            current_iter += 1
            
        # Final
        if buffer: insert_batch(conn, buffer)
        write_progress(total_iterations, total_iterations, "completed")
        logger.info("✅ Historique terminé.")
        
    except Exception as e:
        logger.error(f"Erreur: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    run()
