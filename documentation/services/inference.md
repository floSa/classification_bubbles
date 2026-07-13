# Inference

## Description générale

Moteur de décision temps réel. Le service expose une **API REST FastAPI**, prédit
automatiquement le niveau de bouchage de chaque nouveau spectrogramme (polling MongoDB) et
recharge le modèle à chaud quand Training en produit un nouveau. Service **GPU**.

## Structure du service

| Fichier | Rôle |
|---|---|
| `main.py` | Application FastAPI, tâche de polling, watcher de modèle |
| `utils.py` | `load_model()`, `preprocess_image()` |
| `Dockerfile` | `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` ; `uvicorn main:app --host 0.0.0.0 --port 8000` |
| `requirements.txt` | `fastapi`, `uvicorn[standard]`, `torch>=2.0.0`, `torchvision>=0.15.0`, `pymongo`, `minio`, `Pillow`, `numpy`, `python-dotenv`, `psycopg2-binary` |

## API / interface

| Méthode | Route | Rôle |
|---|---|---|
| `GET` | `/health` | État du service et du modèle (`model_loaded`, `device`) |
| `GET` | `/predict/{bubble_id}` | Prédiction à la demande pour un document MongoDB |

`/predict/{bubble_id}` lit puis **écrit** la prédiction dans MongoDB : le dashboard consomme
ce cache et n'appelle l'API que pour les bulles non encore prédites.

## Polling automatique

Une tâche de fond recherche les bulles à `processed=true`, avec `s3_key` présent et sans
`prediction`, charge l'image depuis MinIO, effectue l'inférence GPU (MobileNetV2) et écrit
le bloc `prediction` (`class`, `label`, `confidence`, `predicted_at`) dans MongoDB.

## Rechargement à chaud

Un watcher surveille le `mtime` de `models/bubble_classifier.pth` (intervalle
`MODEL_WATCH_INTERVAL_S`) et recharge le modèle sans redémarrer le conteneur — utile quand
Inference démarre avant la fin du Training.

## Flux de données

- **Entrée** : images MinIO + métadonnées MongoDB ; modèle `models/bubble_classifier.pth`.
- **Sortie** : champ `prediction` des documents `bubbles` (MongoDB) ; réponses HTTP.

## Lancement

```bash
docker compose up -d inference
curl http://localhost:8000/health
docker compose logs -f inference
```

## Dépendances / port

Dépend de `mongo_db` (`service_started`) et `minio-db` (`service_healthy`). Réservation GPU
NVIDIA, `shm_size: 2gb`. Port **8000** exposé sur l'hôte.
