# Projet AI Tutor

Ce projet est une application web pour un service d'IA qui utilise des modèles de traitement du langage naturel (NLP) pour engager des conversations interactives. Le système comprend une API backend, un modèle d'IA pour la génération de texte, et des outils de transcription audio. Le projet est construit avec **FastAPI** et **Docker** pour une mise en œuvre continue via une pipeline CI/CD.

## Interface de l'application

![Capture d'écran de l'IA](./front/static/assets/images/Screen_20IA.PNG)

## Démo Vidéo

Regardez la démo vidéo de l'application en cliquant sur le lien suivant :

[Voir la démo sur YouTube](https://www.youtube.com/watch?v=FQ0OTguo_iA)

## Fonctionnalités

- **Chat IA** : L'IA engage des conversations naturelles avec les utilisateurs.
- **Transcription audio** : Conversion de l'audio en texte à l'aide de Whisper.
- **Génération de réponses** : Utilisation du modèle **Phi3** via l'API Ollama pour générer des réponses personnalisées.
- **Pipeline CI/CD** : Automatisation des tests et déploiement continu avec Docker et GitHub Actions.
- **Tests automatisés** : Tests unitaires pour vérifier les fonctionnalités principales.

## Technologies utilisées

- **Python 3.11**
- **FastAPI** pour l'API backend
- **Docker** pour la conteneurisation
- **PostgreSQL** pour la base de données
- **NLTK**, **SpaCy** et **Whisper** pour les modèles de traitement du langage naturel
- **Ollama API** pour la génération de texte
- **GitHub Actions** pour CI/CD

## Installation

### Prérequis

- **Docker** et **Docker Compose**
- **Python 3.11** et **pip**

### Étapes d'installation

1. Clonez le dépôt
2. Créez un fichier .env en copiant le fichier .env.example, Modifier et ajouter vos informations de votre basse de donnée.
3. Construisez et démarrez les services Docker : docker-compose up --build
4. Une fois que les conteneurs sont démarrés, vous pouvez accéder à l'application en ouvrant un navigateur et en visitant : http://localhost:8000
5. Pour arrêter les conteneurs Docker, utilisez la commande : docker-compose down

