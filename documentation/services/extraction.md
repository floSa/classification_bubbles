# Extraction

## Description générale

Pont entre les données brutes et les données structurées. Le service surveille TimescaleDB,
détecte les **bursts acoustiques** (bulles) dans le flux et écrit chaque événement découpé
dans MongoDB.

## Structure du service

| Fichier | Rôle |
|---|---|
| `main.py` | Pipeline complet : lecture batch, détection de pics, découpage, insertion |
| `Dockerfile` | `python:3.10-slim` + package `common` |
| `requirements.txt` | `psycopg2-binary`, `numpy`, `scipy`, `pymongo`, `minio`, `python-dotenv` |

Imports partagés : `common.config` (`SAMPLE_RATE`, `DECIMATE_FACTOR`, `MongoConfig`),
`common.db_connections`.

## Paramètres de détection (valeurs lues dans `main.py`)

| Paramètre | Valeur | Rôle |
|---|---|---|
| `BATCH_DURATION` | **1,0 s** | Bloc de traitement (temps réel) |
| `MARGIN_DURATION` | **0,2 s** | Overlap pour éviter les coupures de bulle en bord de bloc |
| `BUBBLE_LENGTH` | **0,2 s** | Durée extraite par bulle |
| `PRE_ROLL` | **0,05 s** | Temps gardé avant le pic (attaque) |
| `PEAK_HEIGHT` | **0,3** | Seuil d'amplitude minimum |
| `MIN_PEAK_DISTANCE` | `0.15 * EFFECTIVE_FS` | Espacement minimum entre pics |

## Pipeline

1. Lecture d'un bloc TimescaleDB (`BATCH_DURATION + MARGIN_DURATION`).
2. Détection des pics via `scipy.signal.find_peaks(height=PEAK_HEIGHT, distance=MIN_PEAK_DISTANCE)`.
3. Découpage de `BUBBLE_LENGTH` autour de chaque pic (avec `PRE_ROLL`).
4. Insertion dans MongoDB (`bubbles`) avec `processed=false`.

## Checkpoint / reprise

Le service persiste sa position dans la collection `state_checkpoints` (curseur
d'extraction), ce qui permet la reprise après redémarrage et évite les doublons. En cas de
gap temporel prolongé, il saute au temps présent.

## Flux de données

- **Entrée** : table `audio_data` (TimescaleDB).
- **Sortie** : collection `bubbles` (MongoDB) — `timestamp`, `label`, `amplitude_max`,
  `duration_sec`, `sample_rate`, `raw_audio`, `processed=false`.

## Lancement

```bash
docker compose up -d extraction
docker compose logs -f extraction
```

## Dépendances / port

Dépend de `timescale_db` (`service_healthy`) et `mongo_db` (`service_started`). Aucun port
exposé.
