# Service Inference - API de prédiction
"""
API REST FastAPI pour l'inférence en temps réel.
Charge le modèle entraîné et prédit le niveau de bouchage
pour les nouvelles bulles détectées.
"""

import os
import io
import time
import logging
import asyncio
import sys
from datetime import datetime
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import MinIOConfig, MongoConfig, INDEX_TO_LABEL
from common.db_connections import get_mongo_collection, get_minio_client
from utils import load_model, preprocess_image, MODELS_DIR, MODEL_FILENAME

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Inference_Service")

# =============================================================================
# VARIABLES GLOBALES
# =============================================================================

model = None
device = None
minio_client = None
mongo_col = None

# =============================================================================
# MODÈLES PYDANTIC
# =============================================================================

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class PredictionResponse(BaseModel):
    bubble_id: str
    predicted_class: int
    predicted_label: str
    confidence: float
    timestamp: str

# =============================================================================
# LIFECYCLE
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application."""
    global model, device, minio_client, mongo_col
    
    logger.info("🚀 Initialisation du service d'inférence...")
    
    # Configuration du device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️ Device: {device}")
    
    # Chargement du modèle
    model = load_model(device)
    if model is None:
        logger.warning("⚠️ Aucun modèle trouvé. Le service fonctionnera en mode dégradé.")
    else:
        logger.info("✅ Modèle chargé avec succès")
    
    # Connexions aux bases
    minio_client = get_minio_client(ensure_bucket=False)
    mongo_col = get_mongo_collection()
    
    # Démarrage du background polling
    # DÉSACTIVÉ: L'inférence se fait à la demande via l'API par Streamlit
    # asyncio.create_task(background_inference_loop())
    
    yield
    
    logger.info("👋 Arrêt du service d'inférence")


app = FastAPI(
    title="Bubble Inference API",
    description="API de prédiction du niveau de bouchage industriel",
    version="1.0.0",
    lifespan=lifespan
)

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérification de l'état du service."""
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        device=str(device) if device else "unknown"
    )


@app.get("/predict/{bubble_id}", response_model=PredictionResponse)
async def predict_single(bubble_id: str):
    """
    Effectue une prédiction pour une bulle spécifique.
    
    Args:
        bubble_id: ID MongoDB de la bulle
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    from bson import ObjectId
    
    try:
        # Récupération du document
        doc = mongo_col.find_one({"_id": ObjectId(bubble_id)})
        if not doc:
            raise HTTPException(status_code=404, detail="Bulle non trouvée")
        
        s3_key = doc.get("s3_key")
        if not s3_key:
            raise HTTPException(status_code=400, detail="Spectrogramme non disponible")
        
        # Chargement et prédiction
        predicted_class, confidence = await asyncio.to_thread(
            predict_from_s3, s3_key
        )
        
        return PredictionResponse(
            bubble_id=bubble_id,
            predicted_class=predicted_class,
            predicted_label=f"{INDEX_TO_LABEL[predicted_class]}%",
            confidence=confidence,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur prédiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# FONCTIONS DE PRÉDICTION
# =============================================================================

def predict_from_s3(s3_key: str) -> tuple[int, float]:
    """
    Charge une image depuis MinIO et effectue la prédiction.
    
    Returns:
        Tuple (classe_prédite, score_confiance)
    """
    # Récupération de l'image
    response = minio_client.get_object(MinIOConfig.BUCKET, s3_key)
    img_data = response.read()
    response.close()
    response.release_conn()
    
    # Prétraitement
    image = Image.open(io.BytesIO(img_data)).convert('L').convert('RGB')
    tensor = preprocess_image(image).unsqueeze(0).to(device)
    
    # Inférence
    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item()

# =============================================================================
# BACKGROUND INFERENCE LOOP
# =============================================================================

async def background_inference_loop():
    """
    Boucle de fond qui poll MongoDB pour traiter automatiquement
    les nouvelles bulles sans prédiction.
    """
    global model  # Déclaration explicite au début
    logger.info("🔄 Démarrage du background polling...")
    
    while True:
        try:
            # Rechargement automatique du modèle si absent
            if model is None:
                # On essaie de recharger
                loaded = load_model(device)
                if loaded:
                    model = loaded
                    logger.info("✅ Modèle détecté et chargé dynamiquement !")
                else:
                    # Toujours pas de modèle, on attend
                    await asyncio.sleep(5)
                    continue
            
            # Recherche des bulles traitées mais sans prédiction
            query = {
                "processed": True,
                "s3_key": {"$exists": True},
                "prediction": {"$exists": False}
            }
            
            cursor = mongo_col.find(query).limit(20)
            count = 0
            
            for doc in cursor:
                try:
                    s3_key = doc.get("s3_key")
                    if not s3_key:
                        continue
                    
                    # Prédiction
                    # predict_from_s3 utilise 'model' qui est global
                    predicted_class, confidence = await asyncio.to_thread(
                        predict_from_s3, s3_key
                    )
                    
                    # Mise à jour MongoDB
                    mongo_col.update_one(
                        {"_id": doc["_id"]},
                        {
                            "$set": {
                                "prediction": {
                                    "class": predicted_class,
                                    "label": INDEX_TO_LABEL[predicted_class],
                                    "confidence": confidence,
                                    "predicted_at": datetime.utcnow()
                                }
                            }
                        }
                    )
                    count += 1
                    
                except Exception as e:
                    logger.error(f"Erreur traitement bulle {doc.get('_id')}: {e}")

            
            if count > 0:
                logger.info(f"🔮 {count} prédictions effectuées")
            
            await asyncio.sleep(2)  # Poll toutes les 2 secondes
            
        except Exception as e:
            logger.error(f"Erreur dans la boucle de background: {e}")
            await asyncio.sleep(5)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
