# Utiliser une image de base avec Python 3.11 et CUDA
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Définir l'environnement pour les installations non interactives
ENV DEBIAN_FRONTEND=noninteractive

# Installer Python 3.11, pip, et d'autres dépendances nécessaires
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
    && rm -rf /var/lib/apt/lists/*

# Installer pip pour Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Créer des liens symboliques pour python et pip
RUN ln -s /usr/bin/python3.11 /usr/bin/python \
    && ln -s /usr/local/bin/pip3.11 /usr/bin/pip

# Définir le fuseau horaire à UTC
RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && dpkg-reconfigure --frontend noninteractive tzdata

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de l'application
COPY . .

# Mettre à jour pip et setuptools
RUN pip install --upgrade pip setuptools

# Copier le fichier requirements.txt
COPY requirements.txt .

# Installer les dépendances à partir de requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Télécharger le modèle spaCy
RUN python3.11 -m spacy download en_core_web_sm

# Copier le fichier .env
COPY .env .

# Exposer le port de l'application
EXPOSE 8000

# Définir la commande par défaut pour démarrer l'application
CMD ["uvicorn", "back.api_main:app", "--host", "0.0.0.0", "--port", "8000"]
