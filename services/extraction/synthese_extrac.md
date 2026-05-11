# Synthèse du Service d'Extraction (synthese_extrac)

## Description Générale
Le service d'extraction surveille TimescaleDB pour détecter les événements acoustiques (bulles) et les enregistrer dans MongoDB. C'est le pont entre les données brutes et les données structurées.

## Structure du Service (v2)

| Fichier | Rôle |
|---------|------|
| `main.py` | Pipeline complet de détection et découpage |
| `Dockerfile` | Image Python 3.10 + package common |
| `requirements.txt` | scipy, pymongo, psycopg2-binary |

### Imports depuis Common
```python
from common.config import SAMPLE_RATE, DECIMATE_FACTOR, MongoConfig
from common.db_connections import get_timescale_connection, get_mongo_client
```

## Algorithme DSP (Traitement du Signal)

### Paramètres de Détection
| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `BATCH_DURATION` | 10s | Taille du bloc de traitement |
| `MARGIN_DURATION` | 0.5s | Overlap pour éviter les coupures |
| `BUBBLE_LENGTH` | 0.2s | Durée extraite par bulle |
| `PRE_ROLL` | 0.05s | Temps gardé avant le pic |
| `PEAK_HEIGHT` | 0.3 | Seuil amplitude minimum |
| `MIN_PEAK_DISTANCE` | ~600 samples | Espacement minimum entre pics |

### Pipeline de Traitement
1. **Lecture batch** depuis TimescaleDB avec marge
2. **Détection des pics** via `scipy.signal.find_peaks()`
3. **Découpage** : 200ms autour de chaque pic
4. **Insertion MongoDB** avec métadonnées

## Système de Checkpoint
Le service utilise MongoDB pour persister sa position dans le flux :
- Collection `state_checkpoints` avec `extraction_cursor`
- Reprise automatique après redémarrage
- Vérification de sécurité pour éviter les doublons

## Document MongoDB Généré
```json
{
    "timestamp": "2026-01-21T12:00:00Z",
    "label": 40,
    "amplitude_max": 0.65,
    "duration_sec": 0.2,
    "sample_rate": 4009,
    "raw_audio": [0.1, -0.2, ...],
    "processed": false,
    "s3_spectrogram_path": null
}
```

## Lancement
```bash
docker compose up extraction
docker compose logs -f extraction
```
