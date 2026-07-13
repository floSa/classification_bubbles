# App (Dashboard)

## Description générale

Dashboard **Streamlit** de visualisation temps réel : signal audio live, spectrogrammes,
progression de la génération de données et prédictions du modèle comparées au label réel.

## Structure du service

| Fichier | Rôle |
|---|---|
| `main.py` | Interface Streamlit et boucle de rafraîchissement |
| `utils.py` | Data fetchers (`@st.cache_resource`), stats de génération et de training |
| `Dockerfile` | `python:3.10-slim` + package `common` |
| `requirements.txt` | `streamlit`, `streamlit-autorefresh`, `pandas`, `psycopg2-binary`, `pymongo`, `minio`, `Pillow`, `python-dotenv`, `requests` |

Imports partagés : `common.config` (`TimescaleConfig`, `MongoConfig`, `MinIOConfig`).

## Composants de l'interface

- **Sidebar** : indicateurs de connexion, contrôle du taux de rafraîchissement (1–10 s),
  toggle auto-refresh.
- **État de l'entraînement** : présence du modèle, taille, dernière modification du `.pth`.
- **Progression de génération** : total / traitées / en attente / erreurs, équilibre des
  classes.
- **Derniers spectrogrammes** : galerie des dernières images MinIO avec label.

Les connexions aux bases utilisent le pattern singleton via `@st.cache_resource`.

## Flux de données

- **Entrée** : TimescaleDB (signal live), MongoDB (événements + prédictions), MinIO
  (spectrogrammes) ; API Inference pour les bulles non encore prédites.
- **Sortie** : interface web (aucune écriture en base).

## Lancement

```bash
docker compose up -d --build app
# puis ouvrir http://localhost:8501
```

## Dépendances / port

Dépend de `timescale_db` (`service_healthy`), `mongo_db` (`service_started`) et `minio-db`
(`service_healthy`). Port **8501** exposé sur l'hôte.
