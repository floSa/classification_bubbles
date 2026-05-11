# Synthèse du Service d'Entraînement (synthese_train)

## Description Générale
Le service d'entraînement crée un modèle de classification CNN sur les spectrogrammes générés. Il utilise **MobileNetV2** avec Transfer Learning pour une convergence rapide.

## Structure du Service (v2)

| Fichier | Rôle |
|---------|------|
| `main.py` | Boucle d'entraînement PyTorch |
| `utils.py` | **Conservé** - `BubbleDataset` et `get_transforms()` |
| `Dockerfile` | **Image PyTorch CUDA** officielle |
| `requirements.txt` | torch, torchvision, pymongo, minio |

### Image Docker
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
```

## Architecture du Modèle

- **Base** : MobileNetV2 pré-entraîné sur ImageNet
- **Head** : Linear layer adapté à 5 classes (0%, 20%, 40%, 60%, 80%)
- **Optimiseur** : Adam (lr=0.001)
- **Loss** : CrossEntropyLoss

## Hyperparamètres

| Paramètre | Valeur |
|-----------|--------|
| `BATCH_SIZE` | 32 |
| `EPOCHS` | 10 |
| `LEARNING_RATE` | 0.001 |
| `VAL_SPLIT` | 0.2 (20% validation) |

## Optimisations GPU

### Mixed Precision Training (AMP)
Utilisation de `torch.amp` pour accélérer l'entraînement sur GPU :

```python
scaler = torch.amp.GradScaler("cuda")

with torch.amp.autocast("cuda"):
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Corrections PyTorch 2.x
- ✅ `torch.amp.autocast("cuda")` au lieu de `torch.cuda.amp.autocast`
- ✅ `torch.amp.GradScaler("cuda")` au lieu de `torch.cuda.amp.GradScaler`
- ✅ `weights=MobileNet_V2_Weights.IMAGENET1K_V1` au lieu de `pretrained=True`

## Data Augmentation

| Transform (Train) | Description |
|-------------------|-------------|
| `RandomHorizontalFlip` | Flip aléatoire |
| `RandomRotation(5)` | Rotation ±5° |
| `Normalize` | ImageNet stats |

## Dataset Custom

Le `BubbleDataset` charge les images directement depuis MinIO :

```python
# Flux de données
MongoDB (métadonnées) → MinIO (images) → Tensor → Model
```

## Persistance du Modèle

- **Volume** : `./models:/app/models`
- **Fichier** : `bubble_classifier.pth`
- **Stratégie** : Sauvegarde du meilleur modèle selon validation accuracy

## Comportement

1. Au démarrage, vérifie si un modèle existe
2. Si oui → Mode veille (sleep)
3. Si non → Entraînement automatique
4. Après entraînement → Mode veille

## Lancement

```bash
# Avec le profil training
docker compose --profile training up training

# Logs
docker compose logs -f training
```
