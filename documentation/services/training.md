# Training

## Description générale

Entraîne un classifieur **MobileNetV2** (transfer learning) sur les spectrogrammes stockés
dans MinIO et sauvegarde le meilleur modèle. Service **GPU**, activé uniquement via le
profil Compose `training`.

## Structure du service

| Fichier | Rôle |
|---|---|
| `main.py` | Boucle d'entraînement PyTorch (AMP, sauvegarde du meilleur modèle) |
| `utils.py` | `BubbleDataset` (chargement des images depuis MinIO), `get_transforms()`, `MODEL_FILENAME` |
| `Dockerfile` | `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` |
| `requirements.txt` | `torch>=2.0.0`, `torchvision>=0.15.0`, `psycopg2-binary`, `pymongo`, `minio`, `Pillow`, `numpy`, `python-dotenv` |

## Modèle et hyperparamètres (valeurs lues dans `main.py`)

| Paramètre | Valeur |
|---|---|
| Architecture | MobileNetV2 (`weights=MobileNet_V2_Weights.IMAGENET1K_V1`) |
| Classes | 5 (0/20/40/60/80 %) |
| `BATCH_SIZE` | 32 |
| `EPOCHS` | 10 |
| `LEARNING_RATE` | 0,001 (Adam) |
| `VAL_SPLIT` | 0,2 |
| Loss | CrossEntropyLoss |
| Mixed precision | `torch.amp.autocast("cuda")` + `torch.amp.GradScaler("cuda")` |

La sauvegarde ne conserve que le **meilleur modèle** selon la précision de validation
(`best_acc`).

## Data augmentation

Voir `utils.py` (`get_transforms`). Le spectrogramme N&B est converti en RGB (MobileNet
attend 3 canaux) puis normalisé aux statistiques ImageNet. Le flip horizontal a été retiré
(il inverserait le temps du signal — voir [audit.md](../../audit.md)).

## Flux de données

- **Entrée** : métadonnées MongoDB (`bubbles`) → images depuis MinIO.
- **Sortie** : `models/bubble_classifier.pth` (volume `./models`).

## Comportement

Au démarrage : si un modèle existe déjà, le service passe en veille ; sinon il entraîne
puis se met en veille.

## Lancement

```bash
docker compose --profile training up -d --build training
docker compose logs -f training
```

## Dépendances / port

Dépend de `mongo_db` (`service_started`) et `minio-db` (`service_healthy`). Réservation GPU
NVIDIA, `shm_size: 2gb`. Aucun port exposé.
