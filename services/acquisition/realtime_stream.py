# Script de flux temps réel
"""
Simule le flux continu APRES l'historique.
"""
import sys
import os
import time
import random
import logging
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import (
    CHUNK_DURATION,
    DECIMATE_FACTOR,
    SAMPLE_RATE,
    CLOGGING_LEVELS,
    LABEL_NOISE_RATE,
)
from common.db_connections import get_timescale_connection
from common.signal_processing import generate_signal, insert_batch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RealTime_Stream")


def _noisy_label(true_level: int) -> int:
    """
    Avec probabilité LABEL_NOISE_RATE, renvoie un label différent du vrai
    niveau utilisé pour la génération (simule des erreurs d'étiquetage).
    """
    if LABEL_NOISE_RATE <= 0:
        return true_level
    if random.random() >= LABEL_NOISE_RATE:
        return true_level
    others = [lv for lv in CLOGGING_LEVELS if lv != true_level]
    return random.choice(others) if others else true_level


def run():
    logger.info("📡 Démarrage Flux Temps Réel...")
    if LABEL_NOISE_RATE > 0:
        logger.info(f"🎲 Bruit de label actif : {LABEL_NOISE_RATE * 100:.1f}% des chunks seront mal étiquetés.")
    conn = get_timescale_connection()

    level_idx = 0
    counter = 0
    SAMPLES_PER_LEVEL = 20

    try:
        while True:
            if counter >= SAMPLES_PER_LEVEL:
                level_idx = (level_idx + 1) % len(CLOGGING_LEVELS)
                counter = 0
                logger.info(f"🔄 Changement régime : {CLOGGING_LEVELS[level_idx]}%")

            level = CLOGGING_LEVELS[level_idx]
            stored_label = _noisy_label(level)
            start_proc = time.time()

            _, signal = generate_signal(level, CHUNK_DURATION)

            now = datetime.now(timezone.utc)
            reduced_sig = signal[::DECIMATE_FACTOR]
            time_step = 1.0 / (SAMPLE_RATE / DECIMATE_FACTOR)
            base_ts = now.timestamp()

            buffer = []
            for i, val in enumerate(reduced_sig):
                ts = datetime.fromtimestamp(base_ts + i * time_step, timezone.utc)
                buffer.append((ts, float(val), int(stored_label)))

            insert_batch(conn, buffer)
            counter += 1

            elapsed = time.time() - start_proc
            time.sleep(max(0, CHUNK_DURATION - elapsed))

    except KeyboardInterrupt:
        pass
    finally:
        conn.close()


if __name__ == "__main__":
    run()
