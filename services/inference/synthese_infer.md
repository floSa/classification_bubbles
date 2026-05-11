# Synthèse du Service d'Inférence (synthese_infer)

## Description Générale
Le service d'inférence est le moteur de décision en temps réel du pipeline. Il expose une **API REST FastAPI** et prédit automatiquement le niveau de bouchage pour chaque nouveau spectrogramme.

## Structure du Service (v2 - NOUVEAU)

| Fichier | Rôle |
|---------|------|
| `main.py` | API FastAPI + Background polling |
| `utils.py` | `load_model()`, `preprocess_image()` |
| `Dockerfile` | Image PyTorch CUDA + uvicorn |
| `requirements.txt` | fastapi, uvicorn, torch, torchvision |

### Image Docker
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## API REST

### Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/health` | État du service et du modèle |
| `GET` | `/predict/{bubble_id}` | Prédiction on-demand par ID MongoDB |

### Exemple de Réponse `/health`
```json
{
    "status": "ok",
    "model_loaded": true,
    "device": "cuda:0"
}
```

### Exemple de Réponse `/predict/{id}`
```json
{
    "bubble_id": "64a1b2c3d4e5f6",
    "predicted_class": 2,
    "predicted_label": "40%",
    "confidence": 0.92,
    "timestamp": "2026-01-21T12:05:00Z"
}
```

## Background Polling

Le service poll automatiquement MongoDB pour traiter les nouvelles bulles :

```python
# Requête MongoDB
query = {
    "processed": True,           # Spectrogramme généré
    "s3_key": {"$exists": True}, # Image disponible
    "prediction": {"$exists": False}  # Pas encore prédit
}
```

### Cycle de Polling
1. Recherche des bulles sans prédiction (limit 20)
2. Chargement image depuis MinIO
3. Prétraitement (Tensor, Normalisation)
4. Inférence GPU (MobileNetV2)
5. Mise à jour MongoDB avec résultat
6. Sleep 2 secondes

## Mise à Jour MongoDB

Après prédiction, le document est enrichi :
```json
{
    "prediction": {
        "class": 2,
        "label": 40,
        "confidence": 0.92,
        "predicted_at": "2026-01-21T12:05:00Z"
    }
}
```

## Chargement du Modèle

Le modèle est chargé au démarrage depuis le volume partagé :
- **Chemin** : `/app/models/bubble_classifier.pth`
- **Architecture** : MobileNetV2 (5 classes)
- **Mode dégradé** : Si pas de modèle, le service retourne 503

## GPU Support

- Device détecté automatiquement (CUDA si disponible)
- Inférence en quelques millisecondes par image
- Réservation GPU dans docker-compose.yml

## Lancement

```bash
# Démarrage (le service démarre automatiquement)
docker compose up -d inference

# Test API
curl http://localhost:8000/health

# Logs
docker compose logs -f inference
```

## Port

**8000** - Exposé et mappé dans docker-compose.yml
