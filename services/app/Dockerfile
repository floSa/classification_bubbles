FROM python:3.10-slim

WORKDIR /app

# Copie du package common
COPY services/common /app/common

# Copie des fichiers du service
COPY services/app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/app/*.py ./
COPY services/app/*.md ./

# Le PYTHONPATH inclut /app pour importer common
ENV PYTHONPATH=/app

# Expose Streamlit port
EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0"]
