# Documentation du Stockage (Storage)

Le Bubble Project utilise une architecture de stockage **polyglotte** pour répondre aux différents besoins de performance et de structure des données.

---

## 1. TimescaleDB (Séries Temporelles)

**Rôle** : Stockage des signaux acoustiques bruts décimés.

| Aspect | Détail |
|--------|--------|
| **Image Docker** | `timescale/timescaledb:latest-pg14` |
| **Port** | 5433 (externe) → 5432 (interne) |
| **Base** | `bubble_db` |
| **Table** | `audio_data` (hypertable) |

### Structure de la Table
```sql
CREATE TABLE audio_data (
    time        TIMESTAMPTZ NOT NULL,
    amplitude   REAL        NOT NULL,
    label       INTEGER     NOT NULL
);
```

### Politique de Rétention
```sql
SELECT add_retention_policy('audio_data', INTERVAL '24 hours');
```

---

## 2. MongoDB (Document / Métadonnées)

**Rôle** : Feature Store pour les événements acoustiques et les prédictions.

| Aspect | Détail |
|--------|--------|
| **Image Docker** | `mongo:latest` |
| **Port** | 27018 (externe) → 27017 (interne) |
| **Database** | `bubble_project` |

### Collections

| Collection | Contenu |
|------------|---------|
| `bubbles` | Événements audio + spectrogrammes + prédictions |
| `extraction_state` | Curseurs de position pour reprise (Gap detection) |

### Document Type (bubbles)
```json
{
    "_id": ObjectId("..."),
    "timestamp": ISODate("2026-01-21T12:00:00Z"),
    "label": 40,
    "amplitude_max": 0.65,
    "duration_sec": 0.2,
    "sample_rate": 4009,
    "raw_audio": [...],
    "processed": true,
    "s3_bucket": "spectrograms",
    "s3_key": "2026/01/21/bubble_xxx.png",
    "s3_url": "http://minio_db:9000/spectrograms/...",
    "prediction": {
        "class": 2,
        "label": 40,
        "confidence": 0.92,
        "predicted_at": ISODate("...")
    }
}
```

---

## 3. MinIO (Stockage d'Objets S3-Compatible)

**Rôle** : Data Lake pour les spectrogrammes PNG.

| Aspect | Détail |
|--------|--------|
| **Image Docker** | `minio/minio:latest` |
| **Port API** | 9000 |
| **Port Console** | 9001 |
| **Bucket** | `spectrograms` |

### Organisation des Fichiers
```
spectrograms/
├── 2026/
│   └── 01/
│       └── 21/
│           ├── bubble_64a1b2c3.png
│           ├── bubble_64a1b2c4.png
│           └── ...
```

### Accès Console
- **URL** : http://localhost:9001
- **User** : `minioadmin`
- **Password** : `minioadmin`

---

## Schéma des Interactions (v2)

```mermaid
graph LR
    subgraph Capteurs
        ACQ[Acquisition]
    end
    
    subgraph Stockage
        TS[(TimescaleDB<br/>Port 5433)]
        MG[(MongoDB<br/>Port 27018)]
        MO[(MinIO<br/>Port 9000)]
    end
    
    subgraph Traitement
        EXT[Extraction]
        TRF[Transformation]
        TRN[Training]
        INF[Inference]
    end
    
    subgraph Visualisation
        APP[Dashboard<br/>Port 8501]
    end
    
    ACQ -->|"audio float16"| TS
    TS -->|"lecture batch"| EXT
    EXT -->|"snippets + labels"| MG
    MG -->|"read unprocessed"| TRF
    TRF -->|"PNG 224×224"| MO
    TRF -->|"update status"| MG
    MO -->|"read images"| TRN
    MO -->|"read images"| INF
    TRN -->|"save model"| MODELS[/models/]
    MODELS -->|"load model"| INF
    INF -->|"write predictions"| MG
    TS -->|"live signal"| APP
    MG -->|"events + preds"| APP
    MO -->|"spectrograms"| APP
```

---

## Volumes Docker

Persistance des données entre redémarrages :

| Volume | Service | Chemin Interne |
|--------|---------|----------------|
| `timescale_data` | timescale_db | `/var/lib/postgresql/data` |
| `mongo_data` | mongo_db | `/data/db` |
| `minio_data` | minio_db | `/data` |
| `./models` | training, inference | `/app/models` |

---

## Chaînes de Connexion

### TimescaleDB
```bash
psql "postgres://postgres:password@localhost:5433/bubble_db"
```

### MongoDB
```bash
mongosh "mongodb://root:password@localhost:27018/"
```

### MinIO (mc client)
```bash
mc alias set myminio http://localhost:9000 minioadmin minioadmin
mc ls myminio/spectrograms/
```
