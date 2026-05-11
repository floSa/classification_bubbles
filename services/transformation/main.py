# Service Transformation - Génération de spectrogrammes
"""
Ce service convertit les événements audio en spectrogrammes PNG
et les stocke dans MinIO pour le training/inference.
"""

import time
import logging
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import MinIOConfig
from common.db_connections import get_mongo_collection, get_minio_client
from utils import create_spectrogram_image, upload_to_minio, compute_fft_for_storage

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Transformation_Service")


def main():
    logger.info("🚀 Démarrage du service Transformation (Spectrogram Generation)...")
    
    col = get_mongo_collection()
    minio_client = get_minio_client()
    
    try:
        while True:
            # 1. Récupérer un lot de bulles non traitées
            cursor = col.find({"processed": False}).limit(50)
            
            count = 0
            for doc in cursor:
                try:
                    mongo_id = str(doc["_id"])
                    raw_audio = doc["raw_audio"]
                    fs = doc["sample_rate"]
                    timestamp_dt = doc["timestamp"]
                    
                    if not raw_audio or len(raw_audio) == 0:
                        logger.warning(f"Audio vide pour {mongo_id}, ignoré.")
                        col.update_one(
                            {"_id": doc["_id"]}, 
                            {"$set": {"processed": "error", "error": "empty_audio"}}
                        )
                        continue

                    # 2. Création de l'image
                    image_stream, db_min, db_max = create_spectrogram_image(raw_audio, fs)
                    
                    # 3. Définition du chemin MinIO (Partitioning par date)
                    date_path = timestamp_dt.strftime("%Y/%m/%d")
                    object_name = f"{date_path}/bubble_{mongo_id}.png"
                    
                    # 4. Upload vers MinIO (avec gestion backpressure)
                    file_length = image_stream.getbuffer().nbytes
                    upload_to_minio(
                        minio_client,
                        MinIOConfig.BUCKET,
                        object_name,
                        image_stream,
                        file_length
                    )
                    
                    # 5. Calcul de la FFT pour stockage
                    fft_data = compute_fft_for_storage(raw_audio, fs)
                    
                    # 6. Mise à jour des métadonnées Mongo
                    col.update_one(
                        {"_id": doc["_id"]},
                        {
                            "$set": {
                                "processed": True,
                                "s3_bucket": MinIOConfig.BUCKET,
                                "s3_key": object_name,
                                "s3_url": f"http://{MinIOConfig.ENDPOINT}/{MinIOConfig.BUCKET}/{object_name}",
                                "processed_at": datetime.utcnow(),
                                "spectro_stats": {"min_db": float(db_min), "max_db": float(db_max)},
                                "fft_data": fft_data
                            }
                        }
                    )
                    count += 1
                
                except Exception as e:
                    logger.error(f"Erreur sur le doc {doc.get('_id')}: {e}")
                    col.update_one(
                        {"_id": doc["_id"]}, 
                        {"$set": {"processed": "error", "error_msg": str(e)}}
                    )
            
            if count > 0:
                logger.info(f"⚡ Transformé et Uploadé {count} spectrogrammes.")
            else:
                time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info("Arrêt du service.")


if __name__ == "__main__":
    main()
