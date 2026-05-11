# Synthèse du Service d'Acquisition (synthese_acqui)

## Description Générale
Le service d'acquisition est le point d'entrée du pipeline de données. Il simule les signaux acoustiques d'une tuyauterie industrielle pour 5 niveaux d'obstruction (0%, 20%, 40%, 60%, 80%) et alimente la base de données temporelle **TimescaleDB**.

## Structure du Service (v2)

Le service utilise désormais le **package `common`** pour les configurations et connexions partagées :

| Fichier | Rôle |
|---------|------|
| `main.py` | Point d'entrée unique (fusion acquisition + init_training) |
| `Dockerfile` | Image Python 3.10 + copie du package common |
| `requirements.txt` | Dépendances minimales |

### Imports depuis Common
```python
from common.config import CHUNK_DURATION, DECIMATE_FACTOR, SAMPLE_RATE, CLOGGING_LEVELS
from common.db_connections import get_timescale_connection
from common.signal_processing import generate_signal, insert_batch, is_db_populated
```

## Spécifications Techniques

### 1. Physique et Échantillonnage
- **Signal Source** : Généré à 44 100 Hz pour une fidélité acoustique maximale.
- **Théorème de Nyquist** : Décimation x11 → ~4009 Hz stocké, suffisant pour bulles à 1200 Hz.
- **Saturation** : À 80% d'obstruction, simulation de cavitation avec écrêtage.

### 2. Paramètres de Signal (BUBBLE_PARAMS)

Le signal est désormais optimisé pour être **clairement visible à tous les niveaux de bouchage** :

| Niveau | Fréquence (Hz) | Intervalle Bulles (s) | Chaos |
|--------|----------------|----------------------|-------|
| 0% | 800 | 0.80 | 0.02 |
| 20% | 850 | 0.70 | 0.05 |
| 40% | 920 | 0.55 | 0.10 |
| 60% | 1050 | 0.40 | 0.15 |
| 80% | 1200 | 0.25 | 0.20 |

**Note** : Le chaos a été réduit (0.20 max au lieu de 0.80) pour garder les bulles visibles même à haut bouchage.

### 3. Bruit de Fond

Formule : `noise_std = 0.01 + (clogging_level / 800.0)`

- À 0% : ~0.01 (quasi-silencieux)
- À 80% : ~0.11 (léger bruit de fond)

Cette réduction garantit que le signal des bulles reste toujours discernable.

### 4. Modes de Fonctionnement

| Mode | Usage | Batch Size | Comportement |
|------|-------|------------|--------------|
| **Turbo** | Initialisation | 50 000 points | Génère 24h d'historique sans pause |
| **Real-time** | Production | Insertion immédiate | Synchronisé sur l'horloge système |

## Flux de Données
- **Entrée** : Paramètres physiques depuis `common.config.BUBBLE_PARAMS`
- **Sortie** : Table `audio_data` dans TimescaleDB (time, amplitude, label)

## Lancement

```bash
# Via Docker Compose
docker compose up acquisition

# Logs
docker compose logs -f acquisition
```

## Dépendances
- `psycopg2-binary` : Connexion PostgreSQL
- `pymongo` : Connexion MongoDB (via package common)
- `minio` : Connexion MinIO (via package common)
- `numpy` : Génération de signal
- `python-dotenv` : Variables d'environnement
