# Acquisition

## Description générale

Point d'entrée du pipeline. Le service simule les signaux acoustiques d'une tuyauterie
industrielle pour **5 niveaux de bouchage** (0/20/40/60/80 %) et alimente TimescaleDB.
Il fonctionne en deux temps : une **population initiale** (turbo) qui remplit l'historique,
puis un **flux temps réel** synchronisé sur l'horloge.

## Structure du service

| Fichier | Rôle |
|---|---|
| `main.py` | Orchestrateur : lance `initial_population` puis `realtime_stream` |
| `initial_population.py` | Remplissage turbo par lots de **50 000** points ; purge conditionnelle (`FORCE_CLEAN`) |
| `realtime_stream.py` | Génération continue synchronisée sur l'horloge système |
| `Dockerfile` | `python:3.10-slim` + copie du package `common` |
| `requirements.txt` | `psycopg2-binary`, `pymongo`, `minio`, `numpy`, `python-dotenv` |

Imports partagés : `common.config` (`SAMPLE_RATE`, `DECIMATE_FACTOR`, `CHUNK_DURATION`,
`CLOGGING_LEVELS`, `BUBBLE_PARAMS`, `LABEL_NOISE_RATE`), `common.signal_processing`
(`generate_signal`, `insert_batch`), `common.db_connections`.

## Modèle de signal

Chaque classe est une **distribution gaussienne** (et non un point fixe) sur la fréquence
fondamentale, l'intervalle inter-bulles, la décroissance, le nombre d'harmoniques,
l'amplitude et le bruit de fond rose. Les distributions de classes adjacentes se
**chevauchent** volontairement. Valeurs réelles dans
[services/common/config.py](../../services/common/config.py) (`BUBBLE_PARAMS`) — par
exemple `freq_mean` de **850 Hz** (classe 0) à **1110 Hz** (classe 80).

- Échantillonnage natif : **44 100 Hz**, décimation **x11** → ~**4009 Hz** stocké.
- Bruit de label optionnel : `LABEL_NOISE_RATE` (défaut `0.0`) stocke volontairement un
  mauvais label sur une fraction des chunks.

## Flux de données

- **Entrée** : `BUBBLE_PARAMS` et constantes de `common.config`.
- **Sortie** : table `audio_data` de TimescaleDB — colonnes `(time, amplitude, label)`.

## Variables d'environnement

| Variable | Rôle | Défaut |
|---|---|---|
| `LABEL_NOISE_RATE` | Fraction de chunks au label volontairement faux | `0.0` |
| `FORCE_CLEAN` | Si `true`, purge `audio_data` et supprime le modèle au démarrage | `false` |
| `TIMESCALE_HOST` / `TIMESCALE_PORT` / `TIMESCALE_USER` / `TIMESCALE_PASSWORD` / `TIMESCALE_DB` | Connexion TimescaleDB | voir `.env.example` |

## Lancement

```bash
docker compose up -d acquisition
docker compose logs -f acquisition
```

## Dépendances / port

Dépend de `timescale_db` (condition `service_healthy`). Aucun port exposé. Monte `./models`
pour la suppression du modèle en mode `FORCE_CLEAN`.
