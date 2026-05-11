# The Bubble Project 🫧

![Démo Temps Réel](demo.gif)

**Architecture Micro-services pour la détection de bouchage industriel via analyse acoustique en temps réel.**

Ce projet simule une usine, génère des données de capteurs (audio), les traite, et les visualise via un **Dashboard Streamlit Temps Réel** avec prédictions ML (MobileNetV2).

## 🏗️ Architecture (10 Services)

### Infrastructure
1.  **TimescaleDB** (Port 5433): Stockage des séries temporelles (Audio brut).
2.  **MongoDB** (Port 27018): Feature Store (Métadonnées événements + prédictions).
3.  **MinIO** (Port 9000/9001): Data Lake (Spectrogrammes).
4.  **MinIO Init**: Script d'initialisation des buckets.

### Services Applicatifs (Python)
5.  **Acquisition**: Générateur de signaux audio simulés (5 niveaux de bouchage).
6.  **Extraction**: Détecte les bursts acoustiques et découpe l'audio.
7.  **Transformation**: Convertit l'audio en Spectrogramme PNG (224×224).
8.  **Training**: Entraîne un modèle MobileNetV2 sur GPU (PyTorch).
9.  **Inference**: API REST (FastAPI) pour prédictions en temps réel.
10. **App**: Dashboard Streamlit de visualisation avec prédictions.

## 🚀 Démarrage Rapide

### Pré-requis
*   **Environnement** : Linux / WSL2 (Ubuntu recommandé).
*   Docker & Docker Compose installés.
    *   Sous WSL2 : Installer [Docker Desktop](https://www.docker.com/products/docker-desktop/) avec l'intégration WSL2 activée.
*   4GB+ de RAM allouée à Docker.
*   **GPU NVIDIA + CUDA** pour Training/Inference (optionnel mais recommandé).
    *   Sous WSL2 : Installer les [drivers NVIDIA pour WSL](https://developer.nvidia.com/cuda/wsl).
*   Un fichier `.env` configuré (voir `.env.example`).

### Variables d'environnement notables
| Variable | Défaut | Effet |
|----------|--------|-------|
| `LABEL_NOISE_RATE` | `0.0` | Probabilité par chunk de stocker un mauvais label (simule un capteur imparfait). 0.03–0.05 = réaliste. |
| `FORCE_CLEAN` | `false` | Si `true` au démarrage de `acquisition`, purge `audio_data` et supprime le modèle entraîné. À n'utiliser que pour repartir de zéro. |

### Configuration initiale (WSL2)
```bash
# Copier le fichier d'environnement
cp .env.example .env

# Vérifier que Docker fonctionne
docker --version
docker compose version
```

### Lancement (sans training)
```bash
docker compose up -d --build
```

### Lancement avec Training
```bash
docker compose --profile training up -d --build
```

*(La première construction peut prendre quelques minutes)*

### Accès
*   **Dashboard Streamlit**: [http://localhost:8501](http://localhost:8501)
*   **API Inference**: [http://localhost:8000/health](http://localhost:8000/health)
*   **MinIO Console**: [http://localhost:9001](http://localhost:9001) (User: `minioadmin`, Pass: `minioadmin`)
*   **MongoDB Compass**: `mongodb://root:password@localhost:27018/`
*   **TimescaleDB**: `postgres://postgres:password@localhost:5433/bubble_db`

## 📂 Structure du Projet
```
.
├── docker-compose.yml       # Orchestration
├── README.md                # Ce fichier
├── STORAGE.md               # Documentation stockage polyglotte
├── audit.md                 # Audit technique du projet
├── requirements-dev.txt     # Dépendances tests
├── init-scripts/            # SQL & Scripts
├── models/                  # Modèles entraînés (volume partagé)
├── tests/                   # Tests unitaires
└── services/
    ├── common/              # Package partagé (config, connexions)
    ├── acquisition/         # Génération de données
    ├── extraction/          # Découpage événements
    ├── transformation/      # DSP & S3 Upload
    ├── training/            # Entraînement ML (GPU)
    ├── inference/           # API FastAPI (GPU)
    └── app/                 # Dashboard Streamlit
```


## 🛠️ Détails Techniques
*   **Package Common**: Code partagé entre services (connexions DB, config, signal processing).
*   **Temps Réel Ultra-Faible Latence**:
    *   Acquisition et Extraction : Batch de **1.0 seconde** pour une fluidité maximale.
    *   Inference : Timeout optimisé (0.5s) et polling à la demande.
*   **Optimisation**: Données audio en `float16`, spectrogrammes 224×224, décimation x11.
*   **Rétention**: TimescaleDB configuré avec politique de 24h.
*   **GPU**: Training et Inference utilisent CUDA via images PyTorch officielles.
*   **Backpressure**: Retry exponentiel sur MinIO pour gérer la surcharge.

## 🧪 Modèle physique de simulation

Les 5 classes (0/20/40/60/80% de bouchage) ne sont **pas** des points fixes : chaque classe est une **distribution stochastique** sur plusieurs paramètres (fréquence fondamentale, intervalle inter-bulles, taux de décroissance, nombre d'harmoniques, amplitude, bruit de fond rose). Les distributions de classes adjacentes se **chevauchent** délibérément.

Conséquence : aucune feature scalaire ne sépare les classes seule. Le modèle doit apprendre une représentation conjointe sur le spectrogramme. Voir [services/common/config.py](services/common/config.py) (`BUBBLE_PARAMS`) et [services/common/signal_processing.py](services/common/signal_processing.py).

Pour rendre le problème encore plus réaliste, activer `LABEL_NOISE_RATE=0.03` dans le `.env`.

## 🧪 Tests
```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## 📊 API Inference

| Endpoint | Description |
|----------|-------------|
| `GET /health` | État du service |
| `GET /predict/{bubble_id}` | Prédiction pour une bulle |

L'API effectue également un polling automatique de MongoDB pour prédire les nouvelles bulles.
