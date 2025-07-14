#!/bin/bash
set -e

echo "🚀 Démarrage de TalkAI..."

# Attendre que PostgreSQL soit prêt
echo "⏳ Attente de PostgreSQL..."
until pg_isready -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER; do
    echo "PostgreSQL n'est pas encore prêt - attente..."
    sleep 2
done

echo "✅ PostgreSQL est prêt!"

# Initialiser la base de données et créer les utilisateurs
echo "🗃️ Initialisation de la base de données et création des utilisateurs..."
python /app/init_users.py

echo "🚀 Démarrage de l'application..."
exec python -m uvicorn back.api_main:app --host 0.0.0.0 --port 8000 --reload