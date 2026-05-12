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

### Configuration initiale (WSL2)
```bash
# Copier le fichier d'environnement
cp .env.example .env

# Vérifier que Docker fonctionne
docker --version
docker compose version
```

### 🚦 Les deux modes de lancement

Le projet a **deux modes** distincts. Choisis le bon selon ce que tu veux faire.

#### Mode 1 — Génération + entraînement (premier démarrage ou ré-entraînement)
```bash
make train
```
Équivalent direct sans `make` :
```bash
docker compose --profile training down -v   # purge volumes (DB + spectros)
rm -f models/*.pth models/*.json             # purge modèle + progress
docker compose --profile training up -d --build
```

Ce que ça fait :
- **Reset complet** : volumes TimescaleDB / MongoDB / MinIO supprimés, modèle effacé.
- **Génère 1h** de données simulées avec le nouveau modèle physique stochastique.
- **Extrait + spectrogramme** chaque bulle.
- **Entraîne MobileNetV2** (10 epochs, ~5-15 min sur GPU).
- Le service `training` détruit automatiquement l'ancien `.pth` avant de relancer un training (sauf si `KEEP_MODEL=true`).

Le dashboard ([localhost:8501](http://localhost:8501)) reste sur la **vue training** (barres de progression) tant que l'entraînement est en cours — il ne bascule en vue temps réel **qu'à la fin** du training, pour ne pas être pollué par les données live qui arrivent en parallèle.

#### Mode 2 — Monitoring uniquement
```bash
make run
```
Équivalent direct :
```bash
docker compose up -d --build
```

Ce que ça fait :
- Lance acquisition / extraction / transformation / inference / dashboard.
- **Ne lance pas le training** (profile `training` désactivé).
- Réutilise les données et le modèle déjà présents.

**Si données ou modèle manquent**, le dashboard affiche un écran "Configuration initiale requise" avec la commande `make train` à exécuter — le service ne tente pas de fonctionner à moitié.

### Autres commandes Makefile

| Commande | Effet |
|----------|-------|
| `make help`   | Liste toutes les commandes disponibles |
| `make stop`   | Stoppe les conteneurs sans toucher aux volumes ni au modèle |
| `make clean`  | `stop` + suppression des volumes et du modèle (état totalement vierge) |
| `make logs`   | Suit `docker logs -f bubble_training` |
| `make status` | Affiche `docker compose ps` + état modèle/progression |

### Variables d'environnement notables
| Variable | Défaut | Effet |
|----------|--------|-------|
| `LABEL_NOISE_RATE` | `0.0` | Probabilité par chunk de stocker un mauvais label (simule un capteur imparfait). 0.03–0.05 = réaliste. |
| `FORCE_CLEAN` | `false` | Si `true`, le service acquisition purge `audio_data` au démarrage. Redondant avec `make train` qui purge les volumes ; utile en cas de redémarrage partiel. |
| `KEEP_MODEL` | `false` | Si `true` côté training, conserve l'ancien `.pth` et passe en mode standby au lieu de ré-entraîner. Pour debug. |

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
