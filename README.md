# The Bubble Project

**Architecture micro-services pour la détection de bouchage industriel via analyse acoustique en temps réel.**

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0%20CUDA%2012.1-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-inference%20API-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)

Le projet simule une usine, génère des données de capteurs (audio), les traite et les
visualise via un dashboard Streamlit temps réel avec prédictions ML (MobileNetV2).

![Démo temps réel](demo.gif)

## Architecture

Pipeline de **10 services** (4 d'infrastructure, 6 applicatifs) orchestrés par Docker
Compose sur un réseau bridge unique.

Infrastructure :
1. **TimescaleDB** (port hôte 5433) : séries temporelles (audio brut).
2. **MongoDB** (port hôte 27018) : feature store (métadonnées + prédictions).
3. **MinIO** (ports 9000/9001) : data lake (spectrogrammes).
4. **MinIO Init** : création du bucket `spectrograms`.

Services applicatifs (Python) :
5. **Acquisition** : génère les signaux audio simulés (5 niveaux de bouchage).
6. **Extraction** : détecte les bursts acoustiques et découpe l'audio.
7. **Transformation** : convertit l'audio en spectrogramme PNG 224×224.
8. **Training** : entraîne un MobileNetV2 sur GPU (PyTorch).
9. **Inference** : API REST FastAPI pour les prédictions temps réel.
10. **App** : dashboard Streamlit de visualisation.

```mermaid
flowchart LR
  subgraph Infra
    ts[(TimescaleDB)]
    mg[(MongoDB)]
    mo[(MinIO)]
  end
  subgraph Traitement
    acq[Acquisition] -->|audio| ts
    ts -->|batch| ext[Extraction]
    ext -->|events| mg
    mg -->|non traités| trf[Transformation]
    trf -->|PNG 224x224| mo
    mo -->|images| inf[Inference]
    inf -->|predictions| mg
  end
  subgraph Visu
    app[Dashboard]
  end
  ts --> app
  mg --> app
  mo --> app
```

> Détails : [documentation/architecture.md](documentation/architecture.md).

## Documentation

| Document | Contenu |
|---|---|
| [architecture.md](documentation/architecture.md) | Services, flux, décisions, réseaux et volumes |
| [SECURITY.md](documentation/SECURITY.md) | Secrets, isolation réseau, dépendances, risques |
| [STORAGE.md](STORAGE.md) | Stockage polyglotte (TimescaleDB, MongoDB, MinIO) |
| [services/](documentation/services/) | Une fiche par service applicatif |
| [audit.md](audit.md) | Audit technique et historique des correctifs |

## Démarrage

**Prérequis**

- Linux / WSL2 (Ubuntu recommandé).
- Docker et Docker Compose. Sous WSL2 : [Docker Desktop](https://www.docker.com/products/docker-desktop/) avec l'intégration WSL2.
- 4 Go+ de RAM alloués à Docker.
- GPU NVIDIA + CUDA pour Training/Inference (optionnel mais recommandé). Sous WSL2 : [drivers NVIDIA pour WSL](https://developer.nvidia.com/cuda/wsl).

**Lancement**

```bash
cp .env.example .env        # renseigner les identifiants
docker compose up -d --build            # sans training
docker compose --profile training up -d --build   # avec le service training
```

La première construction peut prendre quelques minutes.

**Accès**

| Service | URL | Note |
|---|---|---|
| Dashboard Streamlit | http://localhost:8501 | Interface principale |
| API Inference | http://localhost:8000/health | Sonde de vie |
| MinIO Console | http://localhost:9001 | User `minioadmin` / Pass `minioadmin` |
| MongoDB | `mongodb://root:password@localhost:27018/` | Compass |
| TimescaleDB | `postgres://postgres:password@localhost:5433/bubble_db` | psql |

## Configuration

Variables lues dans [.env.example](.env.example). Les identifiants des stores
(`TIMESCALE_*`, `MONGO_*`, `MINIO_*`) y sont définis avec leurs défauts de démo.

| Variable | Défaut | Effet |
|---|---|---|
| `LABEL_NOISE_RATE` | `0.0` | Probabilité par chunk de stocker un mauvais label (capteur imparfait). 0.03–0.05 = réaliste. |
| `FORCE_CLEAN` | `false` | Si `true` au démarrage d'`acquisition`, purge `audio_data` et supprime le modèle. À réserver à un redémarrage à zéro. |

## API / Endpoints

Service `inference` (voir [documentation/services/inference.md](documentation/services/inference.md)).

| Méthode | Route | Rôle |
|---|---|---|
| `GET` | `/health` | État du service et du modèle |
| `GET` | `/predict/{bubble_id}` | Prédiction pour une bulle (met en cache le résultat dans MongoDB) |

L'API effectue aussi un polling automatique de MongoDB pour prédire les nouvelles bulles.

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## Structure du projet

```text
classification_bubbles/
├── docker-compose.yml        # orchestration des 10 services
├── .env.example              # template de configuration
├── README.md                 # ce fichier
├── STORAGE.md                # stockage polyglotte
├── audit.md                  # audit technique
├── requirements-dev.txt      # dépendances des tests
├── init-scripts/             # SQL d'initialisation TimescaleDB
├── models/                   # modèle entraîné (volume partagé)
├── tests/                    # tests unitaires
├── documentation/            # architecture, sécurité, fiches services
└── services/
    ├── common/               # package partagé (config, connexions, signal)
    ├── acquisition/          # génération de signaux
    ├── extraction/           # détection d'événements
    ├── transformation/       # spectrogrammes + upload MinIO
    ├── training/             # entraînement ML (GPU)
    ├── inference/            # API FastAPI (GPU)
    └── app/                  # dashboard Streamlit
```

## Modèle physique de simulation

Les 5 classes (0/20/40/60/80 % de bouchage) ne sont **pas** des points fixes : chaque
classe est une **distribution stochastique** sur plusieurs paramètres (fréquence
fondamentale, intervalle inter-bulles, taux de décroissance, nombre d'harmoniques,
amplitude, bruit de fond rose). Les distributions de classes adjacentes se **chevauchent**
délibérément.

Conséquence : aucune feature scalaire ne sépare les classes seule. Le modèle doit apprendre
une représentation conjointe sur le spectrogramme. Voir
[services/common/config.py](services/common/config.py) (`BUBBLE_PARAMS`) et
[services/common/signal_processing.py](services/common/signal_processing.py). Pour rendre le
problème plus réaliste, activer `LABEL_NOISE_RATE=0.03` dans le `.env`.

## Détails techniques

- **Package `common`** : code partagé entre services (connexions DB, config, traitement du signal).
- **Temps réel** : acquisition et extraction traitent par blocs de **1,0 s** ; l'inférence utilise un timeout court et du polling à la demande.
- **Optimisation** : audio en `float16`, spectrogrammes 224×224, décimation x11.
- **Rétention** : TimescaleDB configuré avec une politique de 24 h.
- **GPU** : Training et Inference utilisent CUDA via les images PyTorch officielles.
- **Backpressure** : retry exponentiel sur MinIO pour absorber la surcharge.

## Licences & composants

Licences des composants tiers réellement utilisés (images Docker et dépendances Python des
services). Une licence non vérifiée avec certitude est marquée « à confirmer ».

### Stores (images Docker)

| Composant | Rôle | Licence |
|---|---|---|
| TimescaleDB (`timescale/timescaledb:latest-pg14`) | Base séries temporelles (audio brut) | Apache 2.0 (core) / Timescale License – TSL (fonctions community) |
| PostgreSQL 14 (base de TimescaleDB) | Moteur SQL sous-jacent | PostgreSQL License (BSD-like) |
| MongoDB (`mongo:latest`) | Feature store (métadonnées + prédictions) | SSPL v1 (Server Side Public License) |
| MinIO (`minio/minio:latest`) | Data lake objet (spectrogrammes) | GNU AGPL v3 |
| MinIO Client `mc` (`minio/mc`) | Init des buckets | GNU AGPL v3 |

### Bibliothèques Python majeures

| Composant | Rôle | Licence |
|---|---|---|
| PyTorch (`torch`) | Entraînement / inférence ML (GPU) | BSD 3-Clause |
| torchvision | Modèle MobileNetV2 + transforms | BSD 3-Clause |
| FastAPI | API REST d'inférence | MIT |
| Uvicorn | Serveur ASGI de l'API | BSD 3-Clause |
| Streamlit | Dashboard temps réel | Apache 2.0 |
| streamlit-autorefresh | Rafraîchissement auto du dashboard | à confirmer (probable MIT) |
| pandas | Manipulation de données (dashboard) | BSD 3-Clause |
| NumPy | Calcul numérique | BSD 3-Clause |
| SciPy | Traitement du signal (extraction/DSP) | BSD 3-Clause |
| Pillow | Génération/lecture des spectrogrammes PNG | MIT-CMU (HPND) |
| psycopg2-binary | Client PostgreSQL/TimescaleDB | LGPL v3 (avec exception OpenSSL) |
| pymongo | Client MongoDB | Apache 2.0 |
| minio (client Python) | Client S3/MinIO | Apache 2.0 |
| requests | Client HTTP (dashboard → API) | Apache 2.0 |
| python-dotenv | Chargement des variables d'environnement | BSD 3-Clause |
| pytest / pytest-cov | Tests unitaires (dev) | MIT |

### Poids du modèle

| Composant | Rôle | Licence |
|---|---|---|
| Poids pré-entraînés MobileNetV2 (torchvision, ImageNet) | Initialisation du modèle | à confirmer (conditions torchvision / ImageNet) |

### Code du projet

MIT — Copyright (c) 2026 floSa. À confirmer : pas de fichier `LICENSE` dans le dépôt.
