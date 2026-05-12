# Service Inference - API de prédiction
"""
API REST FastAPI pour l'inférence en temps réel.
Charge le modèle entraîné et prédit le niveau de bouchage pour les bulles.

Particularités :
- Le modèle est rechargé automatiquement s'il apparaît ou est mis à jour
  après le démarrage du service (utile : training et inference tournent en
  parallèle, le fichier .pth peut arriver après le boot de l'API).
- Les prédictions sont écrites dans MongoDB (cache) pour éviter qu'un client
  ré-interroge l'API à chaque rafraîchissement pour la même bulle.
"""

import os
import io
import logging
import asyncio
import sys
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import MinIOConfig, INDEX_TO_LABEL
from common.db_connections import get_mongo_collection, get_minio_client
from utils import load_model, preprocess_image, MODELS_DIR, MODEL_FILENAME

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Inference_Service")

# =============================================================================
# ÉTAT GLOBAL
# =============================================================================

model = None
model_mtime = 0.0          # mtime du .pth chargé en mémoire
device = None
minio_client = None
mongo_col = None

MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)
MODEL_WATCH_INTERVAL_S = 5  # période de surveillance du .pth


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
# CHARGEMENT / RECHARGEMENT DU MODÈLE
# =============================================================================

def _file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def try_reload_model() -> bool:
    """
    Recharge le modèle si le fichier .pth est apparu ou a été mis à jour
    depuis le dernier chargement. Retourne True si un (re)chargement a eu lieu.
    """
    global model, model_mtime

    current_mtime = _file_mtime(MODEL_PATH)
    if current_mtime == 0.0:
        return False
    if model is not None and current_mtime <= model_mtime:
        return False

    new_model = load_model(device)
    if new_model is None:
        return False

    model = new_model
    model_mtime = current_mtime
    logger.info(f"✅ Modèle (re)chargé depuis {MODEL_PATH} (mtime={current_mtime})")
    return True


async def model_watcher_loop():
    """Vérifie périodiquement si un nouveau modèle est disponible."""
    while True:
        try:
            try_reload_model()
        except Exception as e:
            logger.error(f"Erreur watcher modèle: {e}")
        await asyncio.sleep(MODEL_WATCH_INTERVAL_S)


# =============================================================================
# LIFECYCLE
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global device, minio_client, mongo_col

    logger.info("🚀 Initialisation du service d'inférence...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️ Device: {device}")

    if try_reload_model():
        logger.info("✅ Modèle initial chargé")
    else:
        logger.warning("⚠️ Aucun modèle au démarrage — surveillance activée.")

    minio_client = get_minio_client(ensure_bucket=False)
    mongo_col = get_mongo_collection()

    watcher_task = asyncio.create_task(model_watcher_loop())

    try:
        yield
    finally:
        watcher_task.cancel()
        logger.info("👋 Arrêt du service d'inférence")


app = FastAPI(
    title="Bubble Inference API",
    description="API de prédiction du niveau de bouchage industriel",
    version="1.1.0",
    lifespan=lifespan,
)


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
    )


@app.get("/predict/{bubble_id}", response_model=PredictionResponse)
async def predict_single(bubble_id: str):
    """
    Prédiction pour une bulle donnée.

    Si une prédiction est déjà stockée dans MongoDB on la renvoie sans
    refaire passer l'image dans le modèle (cache implicite).
    """
    if model is None:
        # Tentative de chargement à la volée au cas où le fichier vient d'arriver
        try_reload_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    from bson import ObjectId

    try:
        doc = mongo_col.find_one({"_id": ObjectId(bubble_id)})
        if not doc:
            raise HTTPException(status_code=404, detail="Bulle non trouvée")

        # Cache : si la bulle a déjà été prédite, ne pas recalculer
        cached = doc.get("prediction")
        if cached and "class" in cached and "confidence" in cached:
            return PredictionResponse(
                bubble_id=bubble_id,
                predicted_class=cached["class"],
                predicted_label=f"{INDEX_TO_LABEL[cached['class']]}%",
                confidence=float(cached["confidence"]),
                timestamp=(cached.get("predicted_at") or datetime.now(timezone.utc)).isoformat(),
            )

        s3_key = doc.get("s3_key")
        if not s3_key:
            raise HTTPException(status_code=400, detail="Spectrogramme non disponible")

        predicted_class, confidence = await asyncio.to_thread(predict_from_s3, s3_key)

        now = datetime.now(timezone.utc)
        # Persist en cache pour les prochains appels
        try:
            mongo_col.update_one(
                {"_id": doc["_id"]},
                {"$set": {"prediction": {
                    "class": predicted_class,
                    "label": INDEX_TO_LABEL[predicted_class],
                    "confidence": float(confidence),
                    "predicted_at": now,
                }}},
            )
        except Exception as e:
            logger.warning(f"Impossible de cacher la prédiction pour {bubble_id}: {e}")

        return PredictionResponse(
            bubble_id=bubble_id,
            predicted_class=predicted_class,
            predicted_label=f"{INDEX_TO_LABEL[predicted_class]}%",
            confidence=float(confidence),
            timestamp=now.isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur prédiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# INFÉRENCE
# =============================================================================

def predict_from_s3(s3_key: str) -> tuple[int, float]:
    """Charge un spectrogramme depuis MinIO et effectue la prédiction."""
    response = minio_client.get_object(MinIOConfig.BUCKET, s3_key)
    img_data = response.read()
    response.close()
    response.release_conn()

    image = Image.open(io.BytesIO(img_data)).convert('L').convert('RGB')
    tensor = preprocess_image(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return predicted.item(), confidence.item()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
