# Image PyTorch officielle avec CUDA 12.1
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Copie du package common
COPY services/common /app/common

# Copie des fichiers du service
COPY services/training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/training/*.py ./
COPY services/training/*.md ./

# Le PYTHONPATH inclut /app pour importer common
ENV PYTHONPATH=/app

# Création du dossier models
RUN mkdir -p /app/models

CMD ["python", "main.py"]
