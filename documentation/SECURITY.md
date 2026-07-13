# Sécurité — The Bubble Project

Ce projet est une **démonstration locale** (simulation d'usine, exécution mono-hôte). La
posture ci-dessous décrit ce qui est réellement en place et signale en clair les risques non
traités, acceptables dans ce cadre mais à durcir en production.

## Secrets & configuration

Les secrets (identifiants des trois stores) sont fournis par variables d'environnement via
un fichier `.env` **non versionné**. Le dépôt ne contient que `.env.example` avec des valeurs
de démo. Aucun secret n'est codé en dur dans les services : tout passe par
[services/common/config.py](../services/common/config.py).

| Secret | Où | Rotation |
|---|---|---|
| `TIMESCALE_USER` / `TIMESCALE_PASSWORD` | `.env` (non versionné) | manuelle |
| `MONGO_ROOT_USER` / `MONGO_ROOT_PASSWORD` | `.env` (non versionné) | manuelle |
| `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` | `.env` (non versionné) | manuelle |

> **Important** : `.env` ne doit jamais être committé. Il est couvert par
> [.gitignore](../.gitignore). Les valeurs de `.env.example` (`postgres/password`,
> `root/password`, `minioadmin/minioadmin`) sont des identifiants de démo — à remplacer pour
> toute exposition hors machine locale.

## Isolation réseau

Tous les services partagent le réseau bridge `bubble_net` et se joignent par nom DNS Docker.
Seuls les ports utiles au développement sont mappés sur l'hôte.

| Service | Exposé à l'hôte ? | Détail |
|---|---|---|
| `timescale_db` | oui | `5433` (défaut `.env`) — confort de dev (client SQL) |
| `mongo_db` | oui | `27018` (défaut `.env`) — confort de dev (Compass) |
| `minio-db` | oui | `9000` (API) / `9001` (console) |
| `inference` | oui | `8000` (API REST) |
| `app` | oui | `8501` (dashboard) |
| `acquisition`, `extraction`, `transformation`, `training` | non | accès réseau interne uniquement |

## Bucket MinIO

Le bucket `spectrograms` est créé par `minio_init` puis passé en **accès anonyme public**
(`mc anonymous set public`). C'est volontaire pour que le dashboard affiche les images par
URL directe.

> **Risque** : toute personne ayant accès au port `9000` peut lire les spectrogrammes sans
> authentification. Acceptable en local ; en production, servir les images via URLs
> pré-signées plutôt qu'un bucket public.

## Dépendances

Les dépendances Python sont déclarées dans les `requirements.txt` par service mais **non
épinglées** (sauf `torch>=2.0.0` / `torchvision>=0.15.0`). Aucun lockfile ni audit
automatisé n'est en place.

> **Risque** : builds non reproductibles et absence de veille CVE. Piste : figer les versions
> et ajouter un audit (`pip-audit`).

## Conteneurs

Les services CPU partent de `python:3.10-slim`, les services ML de l'image officielle
`pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`. Aucun durcissement supplémentaire :

> **Risque** : les conteneurs tournent en **root** (utilisateur par défaut des images). En
> production, définir un utilisateur non privilégié.

## Données & accès

Les données sont entièrement **synthétiques** (signaux simulés) : aucune donnée personnelle
ni sensible. TimescaleDB applique une rétention de 24 h sur `audio_data`
([STORAGE.md](../STORAGE.md)). Le volume `./models` contient le modèle entraîné, non sensible.

## Risques connus (récapitulatif)

- **Identifiants de démo faibles** dans `.env.example` — à remplacer hors usage local.
- **Bucket public** en lecture anonyme — à restreindre en production.
- **Conteneurs root** et **dépendances non épinglées** — durcissements non appliqués.
- Aucun secret manager, aucun chiffrement au repos au-delà des défauts des images.
