# Service Training - Entraînement du modèle CNN
"""
Service d'entraînement du classificateur de bulles.
Utilise MobileNetV2 avec Transfer Learning.
"""

import os
import time
import json
import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import LABEL_TO_INDEX, MongoConfig
from common.db_connections import get_timescale_connection, get_mongo_collection
from utils import BubbleDataset, get_transforms, MODELS_DIR, MODEL_FILENAME

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Training_Service")

# =============================================================================
# HYPERPARAMÈTRES
# =============================================================================

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2
NUM_CLASSES = len(LABEL_TO_INDEX)

# =============================================================================
# FONCTIONS D'ENTRAÎNEMENT
# =============================================================================

# Seuils minimum pour l'entraînement
MIN_SPECTROGRAMS = 50  # Nombre minimum de spectrogrammes traités
WAIT_INTERVAL = 10     # Secondes entre les vérifications


def check_model_exists():
    """Vérifie si un modèle existe déjà dans le volume dédié."""
    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    return os.path.exists(model_path)


def check_timescale_has_data() -> bool:
    """Vérifie si TimescaleDB contient des données brutes."""
    try:
        conn = get_timescale_connection(max_retries=3, retry_delay=2.0)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM audio_data LIMIT 1;")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count > 0
    except Exception as e:
        logger.warning(f"⚠️ Erreur vérification TimescaleDB: {e}")
        return False


def count_processed_spectrograms() -> int:
    """Compte le nombre de spectrogrammes traités dans MongoDB."""
    try:
        col = get_mongo_collection()
        return col.count_documents({"processed": True, "s3_key": {"$exists": True}})
    except Exception as e:
        logger.warning(f"⚠️ Erreur comptage MongoDB: {e}")
        return 0


# Chemin du fichier de progression
PROGRESS_FILE = os.path.join(MODELS_DIR, "training_progress.json")
ACQ_PROGRESS_FILE = os.path.join(MODELS_DIR, "acquisition_progress.json")


def update_training_progress(status="training", epoch=0, total_epochs=0, loss=0.0, message=""):
    """Écrit la progression dans le fichier JSON partagé."""
    progress = {
        "status": status,
        "current_epoch": epoch,
        "total_epochs": total_epochs,
        "loss": loss,
        "message": message,
        "timestamp": time.time()
    }
    try:
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress, f)
    except Exception as e:
        logger.warning(f"Erreur écriture progression: {e}")


def get_acquisition_status():
    """Lit le statut de l'acquisition."""
    if os.path.exists(ACQ_PROGRESS_FILE):
        try:
            with open(ACQ_PROGRESS_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {}


def wait_for_data_generation():
    """
    LOGIQUE STRICTE : 
    1. Attend que l'Acquisition soit TERMINEE (génération 1h finie).
    2. Attend que la Transformation ait traité TOUS les chunks générés.
    """
    logger.info("⏳ Phase 1 : Attente de la fin de l'Acquisition...")
    
    # Etape 1 : Attente fin Acquisition
    total_generated = 0
    while True:
        acq_status = get_acquisition_status()
        status = acq_status.get("status", "unknown")
        current = acq_status.get("current", 0)
        total = acq_status.get("total", 0)
        
        update_training_progress(
            status="waiting_data", 
            message=f"Acquisition en cours : {current}/{total} chunks"
        )
        
        if status == "completed":
            total_generated = total
            logger.info(f"✅ Acquisition terminée ({total_generated} chunks).")
            break
            
        time.sleep(WAIT_INTERVAL)

    # Etape 2 : Attente fin Transformation (Spectrogrammes)
    # On tolère une petite marge d'erreur (ex: 95% des données) car il peut y avoir des pertes (bout de signal coupé)
    target_spectrograms = int(total_generated * 0.95) 
    
    logger.info(f"⏳ Phase 2 : Attente de la Transformation ({target_spectrograms} spectrogrammes requis)...")
    
    while True:
        count = count_processed_spectrograms()
        
        update_training_progress(
            status="waiting_data", 
            message=f"Transformation en cours : {count}/{target_spectrograms} spectrogrammes (Cible atteinte à 95%)"
        )
        
        if count >= target_spectrograms:
            logger.info(f"✅ {count} spectrogrammes prêts (Target: {target_spectrograms}). Lancement Training.")
            update_training_progress(status="starting", message="Démarrage de l'entraînement...")
            return
            
        logger.info(f"📊 Transformation: {count}/{target_spectrograms}...")
        time.sleep(WAIT_INTERVAL)



def train_model():
    """Lance la boucle d'entraînement optimisée."""
    logger.info("🚀 Démarrage de l'entraînement...")
    update_training_progress(status="starting", message="Initialisation du modèle...")

    
    # 1. Préparation des Données
    full_dataset = BubbleDataset(transform=get_transforms(train=True))
    dataset_size = len(full_dataset)
    
    if dataset_size == 0:
        logger.warning("⚠️ Aucune donnée trouvée dans la base. Annulation.")
        return False

    val_size = int(dataset_size * VAL_SPLIT)
    train_size = dataset_size - val_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    # Création de datasets séparés pour train et val avec les bonnes transforms
    train_loader = DataLoader(
        train_data, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    logger.info(f"📊 Données: {train_size} train, {val_size} validation.")

    # 2. Configuration du Modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️ Utilisation du device : {device}")

    # MobileNetV2 avec weights modernes (PyTorch 2.x compatible)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # Adaptation de la dernière couche pour notre nombre de classes
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # 3. Boucle d'entraînement
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass avec AMP
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # Backward pass avec scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / train_size
        epoch_acc = correct / total
        
        # Validation
        val_acc = evaluate(model, val_loader, device)
        
        logger.info(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Train Acc: {epoch_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )
        
        # Écrire la progression dans un fichier JSON
        progress_data = {
            "current_epoch": epoch + 1,
            "total_epochs": EPOCHS,
            "loss": epoch_loss,
            "train_acc": epoch_acc,
            "val_acc": val_acc,
            "status": "training"
        }
        progress_file = os.path.join(MODELS_DIR, "training_progress.json")
        with open(progress_file, "w") as f:
            json.dump(progress_data, f)
        
        # Sauvegarde du meilleur modèle
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
            torch.save(model.state_dict(), save_path)
            logger.info(f"💾 Modèle sauvegardé (Meilleur Val Acc: {best_acc:.4f})")

    # Marquer l'entraînement comme terminé
    progress_data = {
        "current_epoch": EPOCHS,
        "total_epochs": EPOCHS,
        "status": "completed",
        "train_acc": epoch_acc,
        "val_acc": val_acc
    }
    progress_file = os.path.join(MODELS_DIR, "training_progress.json")
    with open(progress_file, "w") as f:
        json.dump(progress_data, f)
    
    logger.info("🏁 Entraînement terminé.")
    return True


def evaluate(model, loader, device):
    """Évalue le modèle sur un dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0


def main():
    logger.info("🔧 Service d'Entraînement Initialisé.")
    
    # Vérification dossier models
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)
        logger.info(f"📁 Dossier {MODELS_DIR} créé.")

    # 1. Vérifier si un modèle existe déjà
    if check_model_exists():
        logger.info(f"✅ Un modèle existe déjà ({MODEL_FILENAME}).")
        logger.info("Le service passe en mode attente. Relancez manuellement pour forcer le ré-entraînement.")
        while True:
            time.sleep(3600)
    
    # 2. Vérifier si TimescaleDB contient des données brutes
    logger.info("🔍 Vérification des données brutes dans TimescaleDB...")
    has_raw_data = check_timescale_has_data()
    
    if has_raw_data:
        logger.info("✅ Données brutes trouvées dans TimescaleDB.")
    else:
        logger.info("⚠️ Aucune donnée dans TimescaleDB. Attente de la génération...")
    
    # 3. Attendre suffisamment de spectrogrammes traités
    wait_for_data_generation()
    
    # 4. Lancer l'entraînement
    logger.info("🚀 Lancement de l'entraînement...")
    success = train_model()
    
    if success:
        logger.info("✅ Entraînement terminé avec succès. Passage en veille.")
    else:
        logger.warning("⚠️ Entraînement échoué. Retry dans 5 min...")
    
    # Mode veille
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
