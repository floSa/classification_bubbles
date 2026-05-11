# Synthèse du Service de Transformation (synthese_transfo)

## Description Générale
Le service de transformation convertit les événements audio bruts en spectrogrammes PNG optimisés pour l'apprentissage machine. Il gère également l'upload vers MinIO avec gestion du backpressure.

## Structure du Service (v2)

| Fichier | Rôle |
|---------|------|
| `main.py` | Boucle de traitement principale |
| `utils.py` | **Conservé** - Fonctions spécifiques (spectrogramme, upload) |
| `Dockerfile` | Image Python 3.10 + package common |
| `requirements.txt` | scipy, Pillow, minio |

### Imports
```python
from common.config import MinIOConfig
from common.db_connections import get_mongo_collection, get_minio_client
from utils import create_spectrogram_image, upload_to_minio  # Local
```

## Pipeline de Transformation

```
raw_audio (list) 
    ↓ scipy.signal.spectrogram (nperseg=256, noverlap=128)
    ↓ 10×log₁₀(Sxx + ε) → Conversion dB
    ↓ Min-Max Normalisation → 0-255
    ↓ Flip vertical (basses fréquences en bas)
    ↓ Pillow Image mode 'L' (Grayscale)
    ↓ Resize 224×224 (LANCZOS)
    ↓ io.BytesIO (PNG en mémoire)
    ↓ MinIO Upload
```

## Gestion du Backpressure

Le service implémente un **retry avec backoff exponentiel** pour gérer la surcharge MinIO :

```python
@retry_with_backoff(max_attempts=3, initial_delay=1.0, max_delay=10.0)
def upload_to_minio(client, bucket, object_name, data, length):
    ...
```

| Tentative | Délai |
|-----------|-------|
| 1 | 1s |
| 2 | 2s |
| 3 | 4s (max 10s) |

## Organisation des Fichiers MinIO

Partitionnement temporel pour faciliter les requêtes :
```
spectrograms/
├── 2026/
│   └── 01/
│       └── 21/
│           ├── bubble_64a1b2c3d4e5f6.png
│           ├── bubble_64a1b2c3d4e5f7.png
│           └── ...
```

## Métadonnées Ajoutées à MongoDB

Après traitement, le document est enrichi :
```json
{
    "processed": true,
    "s3_bucket": "spectrograms",
    "s3_key": "2026/01/21/bubble_xxx.png",
    "s3_url": "http://minio_db:9000/spectrograms/...",
    "processed_at": "2026-01-21T12:01:00Z",
    "spectro_stats": {"min_db": -80.5, "max_db": -10.2}
}
```

## Lancement
```bash
docker compose up transformation
docker compose logs -f transformation
```
