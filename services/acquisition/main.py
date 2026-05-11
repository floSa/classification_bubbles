# Main Orchestrator
"""
Orchestre l'acquisition :
1. Vérifie si des données existent déjà (skip génération si oui)
2. Lance la population initiale si nécessaire
3. Lance le stream temps réel
"""
import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from initial_population import run as run_init
from realtime_stream import run as run_stream
from common.db_connections import get_timescale_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Acquisition_Main")

# Chemins des fichiers de progression
PROGRESS_DIR = "/app/models"
ACQ_FILE = os.path.join(PROGRESS_DIR, "acquisition_progress.json")


def check_data_exists() -> bool:
    """
    Vérifie si TimescaleDB contient déjà des données.
    
    Returns:
        True si des données existent, False sinon
    """
    try:
        conn = get_timescale_connection(max_retries=5, retry_delay=2.0)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM audio_data LIMIT 1;")
            count = cur.fetchone()[0]
        conn.close()
        return count > 0
    except Exception as e:
        logger.warning(f"⚠️ Erreur vérification DB: {e}")
        return False


def set_skip_mode_progress():
    """
    Quand on skip la génération (données existent déjà),
    on marque l'acquisition comme 'completed' pour que le Dashboard
    passe directement à l'étape suivante.
    """
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    
    # Marquer acquisition comme terminée
    acq_data = {
        "current": 1, "total": 1,
        "percentage": 100.0,
        "status": "completed"
    }
    with open(ACQ_FILE, "w") as f:
        json.dump(acq_data, f)
    logger.info("📝 Acquisition marquée comme 'completed' (mode skip)")


if __name__ == "__main__":
    logger.info("🎬 Démarrage Service Acquisition")
    
    # 1. Vérifier si des données existent déjà
    if check_data_exists():
        logger.info("✅ Données existantes détectées dans TimescaleDB.")
        logger.info("🔄 Skip de la génération historique. Passage direct au flux temps réel.")
        set_skip_mode_progress()
    else:
        # Phase Historique seulement si DB vide
        logger.info("📦 Aucune donnée détectée. Lancement de la génération historique...")
        try:
            run_init()
        except Exception as e:
            logger.error(f"❌ Erreur Initiale: {e}")
            exit(1)
        
    # 2. Phase Temps Réel
    try:
        run_stream()
    except Exception as e:
        logger.error(f"❌ Erreur Stream: {e}")
