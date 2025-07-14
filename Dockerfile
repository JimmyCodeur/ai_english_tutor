FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-distutils \
    python3.11-venv \
    python3.11-dev \
    libsndfile1 \
    ffmpeg \
    tzdata \
    curl \
    build-essential \
    libpq-dev \
    postgresql-client \
    espeak-ng \
    espeak-ng-data \
    && rm -rf /var/lib/apt/lists/*

# Installer pip pour Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Créer des liens symboliques
RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Configurer le timezone
RUN ln -fs /usr/share/zoneinfo/Europe/Paris /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

WORKDIR /app

# Copier les requirements en premier pour le cache Docker
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Télécharger le modèle spaCy
RUN python -m spacy download en_core_web_sm

# Copier le code de l'application
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p /app/front/static/assets/images/avatars \
    /app/audio/user \
    /app/audio/ia \
    /app/logs

# Copier le script d'initialisation des utilisateurs
COPY init_users.py /app/init_users.py

# Copier et rendre exécutables les scripts
COPY wait-for-postgres.sh /wait-for-postgres.sh
COPY wait-and-init.sh /wait-and-init.sh
RUN chmod +x /wait-for-postgres.sh /wait-and-init.sh

EXPOSE 8000