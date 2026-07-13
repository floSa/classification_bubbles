# Transformation

## Description générale

Convertit les événements audio bruts en **spectrogrammes PNG 224×224** prêts pour le CNN,
et les envoie sur MinIO. Gère la surcharge de MinIO par un **retry à backoff exponentiel**.

## Structure du service

| Fichier | Rôle |
|---|---|
| `main.py` | Boucle de traitement des bulles non traitées |
| `utils.py` | `create_spectrogram_image`, `upload_to_minio`, décorateur `retry_with_backoff`, `compute_fft_for_storage` |
| `Dockerfile` | `python:3.10-slim` + package `common` |
| `requirements.txt` | `psycopg2-binary`, `numpy`, `scipy`, `Pillow`, `pymongo`, `minio`, `python-dotenv` |

Imports partagés : `common.config` (`IMG_SIZE`, `MinIOConfig`), `common.db_connections`.

## Pipeline (valeurs lues dans `utils.py`)

1. `scipy.signal.spectrogram(signal, fs, nperseg=256, noverlap=128)`.
2. Conversion en dB (`10·log10(Sxx + ε)`) puis normalisation min-max sur 0–255.
3. Flip vertical (basses fréquences en bas), image Pillow mode `L`.
4. Redimensionnement à `IMG_SIZE` = **(224, 224)** avec `Image.Resampling.LANCZOS`.
5. Encodage PNG en mémoire (`io.BytesIO`) et upload MinIO.

## Backpressure

`retry_with_backoff(max_attempts=3, initial_delay=1.0, max_delay=10.0)` : délais 1 s → 2 s →
4 s (plafonné à 10 s) pour absorber une surcharge de MinIO.

## Flux de données

- **Entrée** : documents `bubbles` avec `processed=false` (MongoDB).
- **Sortie** : PNG dans le bucket `spectrograms` (MinIO), partitionné `AAAA/MM/JJ/bubble_<id>.png` ;
  document MongoDB enrichi (`processed=true`, `s3_bucket`, `s3_key`, `s3_url`, `processed_at`,
  `spectro_stats`).

## Lancement

```bash
docker compose up -d transformation
docker compose logs -f transformation
```

## Dépendances / port

Dépend de `mongo_db` (`service_started`) et `minio-db` (`service_healthy`). Aucun port
exposé.
