# Image PyTorch officielle avec CUDA 12.1
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Copie du package common
COPY services/common /app/common

# Copie des fichiers du service
COPY services/inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/inference/*.py ./
COPY services/inference/*.md ./

# Le PYTHONPATH inclut /app pour importer common
ENV PYTHONPATH=/app

# Port FastAPI
EXPOSE 8000

# Création du dossier models (sera monté en volume)
RUN mkdir -p /app/models

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
