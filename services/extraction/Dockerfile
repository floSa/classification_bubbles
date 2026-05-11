FROM python:3.10-slim

WORKDIR /app

# Copie du package common
COPY services/common /app/common

# Copie des fichiers du service
COPY services/extraction/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/extraction/*.py ./
COPY services/extraction/*.md ./

# Le PYTHONPATH inclut /app pour importer common
ENV PYTHONPATH=/app

CMD ["python", "main.py"]
