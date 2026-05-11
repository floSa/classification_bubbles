# Utilitaires du Service Training
"""
Dataset PyTorch personnalisé pour charger les spectrogrammes
depuis MongoDB + MinIO.
"""

import os
import io
import logging
import sys
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import MinIOConfig, MongoConfig, LABEL_TO_INDEX
from common.db_connections import get_mongo_collection, get_minio_client

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS_DIR = "/app/models"
MODEL_FILENAME = "bubble_classifier.pth"

# =============================================================================
# DATASET PERSONNALISÉ
# =============================================================================

class BubbleDataset(Dataset):
    """
    Dataset optimisé pour récupérer les spectrogrammes de MinIO.
    """
    def __init__(self, transform=None):
        self.logger = logging.getLogger("BubbleDataset")
        self.transform = transform
        self.col = get_mongo_collection()
        self.minio_client = get_minio_client(ensure_bucket=False)
        
        # On charge uniquement les métadonnées (IDs + Labels) en mémoire
        query = {
            "processed": True, 
            "label": {"$exists": True}, 
            "s3_key": {"$exists": True}
        }
        self.samples = list(self.col.find(query, {"_id": 0, "s3_key": 1, "label": 1}))
        
        self.logger.info(f"📊 Dataset chargé : {len(self.samples)} échantillons.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        object_name = item["s3_key"]
        label_val = item["label"]
        
        # Récupération de l'image depuis MinIO
        try:
            response = self.minio_client.get_object(MinIOConfig.BUCKET, object_name)
            img_data = response.read()
            response.close()
            response.release_conn()
            
            # Spectro en N&B -> RGB (MobileNet attend 3 canaux)
            image = Image.open(io.BytesIO(img_data)).convert('L')
            image = image.convert("RGB")
            
            if self.transform:
                image = self.transform(image)
                
            # Conversion du label (0, 20, 40, 60, 80) -> (0, 1, 2, 3, 4)
            label = LABEL_TO_INDEX.get(label_val, 0)
            
            return image, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur chargement {object_name}: {e}")
            # Retourner une image noire pour éviter de casser le batch
            return torch.zeros((3, 224, 224)), torch.tensor(0)

# =============================================================================
# TRANSFORMATIONS
# =============================================================================

def get_transforms(train: bool = True):
    """
    Retourne les transformations pour train ou validation.
    Les images sont déjà redimensionnées en 224x224 par le service Transformation.
    """
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Data Augmentation
            transforms.RandomRotation(5),        # Légère rotation
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
