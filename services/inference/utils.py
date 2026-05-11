# Utilitaires du Service Inference
"""
Fonctions de chargement du modèle et prétraitement des images.
"""

import os
import logging
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import LABEL_TO_INDEX

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Inference_Utils")

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS_DIR = "/app/models"
MODEL_FILENAME = "bubble_classifier.pth"
NUM_CLASSES = len(LABEL_TO_INDEX)

# =============================================================================
# CHARGEMENT DU MODÈLE
# =============================================================================

def load_model(device: torch.device):
    """
    Charge le modèle MobileNetV2 depuis le volume partagé.
    
    Args:
        device: torch.device sur lequel charger le modèle
    
    Returns:
        Modèle chargé ou None si le fichier n'existe pas
    """
    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    
    if not os.path.exists(model_path):
        logger.warning(f"⚠️ Fichier modèle non trouvé: {model_path}")
        return None
    
    try:
        # Création de l'architecture
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
        
        # Chargement des poids
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        
        logger.info(f"✅ Modèle chargé depuis {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle: {e}")
        return None

# =============================================================================
# PRÉTRAITEMENT
# =============================================================================

# Transform identique à celui utilisé pour le training (sans augmentation)
_inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_image(image):
    """
    Prétraite une image PIL pour l'inférence.
    
    Args:
        image: Image PIL en mode RGB
    
    Returns:
        Tensor normalisé
    """
    return _inference_transform(image)
