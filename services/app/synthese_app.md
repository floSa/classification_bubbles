# Synthèse du Service App - Dashboard (synthese_app)

## Description Générale
Le dashboard Streamlit fournit une interface de **suivi de l'entraînement** du modèle de classification des bulles. Il affiche la progression de la génération de données et l'état du modèle.

## Structure du Service (v2)

| Fichier | Rôle |
|---------|------|
| `main.py` | Interface Streamlit - Mode Training |
| `utils.py` | Data fetchers avec `@st.cache_resource` + fonctions de stats training |
| `Dockerfile` | Image Python 3.10 + package common |
| `requirements.txt` | streamlit, pandas, psycopg2, pymongo, minio |

### Imports depuis Common
```python
from common.config import TimescaleConfig, MongoConfig, MinIOConfig
```

## Composants de l'Interface (Mode Training)

### 1. Sidebar
- Indicateurs de connexion (MongoDB ✅, MinIO ✅)
- Contrôle du taux de rafraîchissement (1s - 10s)
- Toggle Auto-Refresh

### 2. État de l'Entraînement
- **Status** : En attente / Modèle Entraîné
- **Taille du Modèle** : En MB si disponible
- **Dernière Mise à Jour** : Timestamp du fichier `.pth`

### 3. Progression Génération de Données
- Compteurs : Total / Traitées / En Attente / Erreurs
- Barre de progression
- **Équilibre des Classes** : Bar chart de la répartition par niveau de bouchage

### 4. Derniers Spectrogrammes
- Galerie des 4 derniers spectrogrammes générés
- Timestamp et label pour chaque image

## Connexions aux Bases

Utilisation du pattern Singleton via `@st.cache_resource` :

```python
@st.cache_resource
def get_mongo_client():
    return MongoClient(MongoConfig.get_uri())

@st.cache_resource
def get_minio_client():
    return Minio(MinIOConfig.ENDPOINT, ...)
```

## Fonctions de Suivi Training

```python
def fetch_data_generation_stats(mongo_client) -> dict:
    """Retourne total, processed, pending, errors, by_label"""

def fetch_training_status() -> dict:
    """Vérifie si le modèle existe et retourne model_exists, model_size, last_modified"""
```

## Boucle de Rafraîchissement

```python
if auto_refresh:
    while True:
        render_dashboard()
        time.sleep(refresh_rate)  # 3s par défaut
```

## Lancement

```bash
docker compose --profile training up -d --build
# Ouvrir http://localhost:8501
```

## Port

**8501** - Port standard Streamlit
