# ════════════════════════════════════════════════════════════
# RAG-LLM Multi-Agent — Dockerfile
# ════════════════════════════════════════════════════════════
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Master IA & Big Data"
LABEL description="RAG-LLM Multi-Agent System"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FLASK_APP=app/app.py \
    FLASK_ENV=production

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Répertoire de travail
WORKDIR /app

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

# Créer les répertoires de données
RUN mkdir -p data/raw_pdfs data/processed data/evaluation models vectorstore

# Exposer le port Flask
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Commande de démarrage
CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "300", "app.app:create_app()"]
