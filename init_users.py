#!/usr/bin/env python3
import sys
import os
sys.path.append('/app')

from back.bdd.database import init_db, SessionLocal, engine
from sqlalchemy import text
from passlib.context import CryptContext
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_default_users():
    """Créer les utilisateurs par défaut avec SQL direct"""
    try:
        # Initialiser les tables
        init_db()
        logger.info('✅ Tables initialisées')
        
        pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')
        
        # Hash des mots de passe
        test_password_hash = pwd_context.hash('password')
        admin_password_hash = pwd_context.hash('admin123')
        
        # Créer les utilisateurs avec SQL direct et transaction
        with engine.begin() as conn:  # ← UTILISER begin() pour auto-commit
            # Vérifier si les utilisateurs existent
            test_exists = conn.execute(text("SELECT COUNT(*) FROM users WHERE email = 'test@talkai.com'")).scalar()
            admin_exists = conn.execute(text("SELECT COUNT(*) FROM users WHERE email = 'admin@talkai.com'")).scalar()
            
            if test_exists == 0:
                conn.execute(text("""
                    INSERT INTO users (email, hashed_password, nom, date_naissance, consent, role, bio, country)
                    VALUES (:email, :password, :nom, :date_naissance, :consent, :role, :bio, :country)
                """), {
                    'email': 'test@talkai.com',
                    'password': test_password_hash,
                    'nom': 'Marc Testeur',
                    'date_naissance': '1990-05-15',
                    'consent': True,
                    'role': 'utilisateur',
                    'bio': 'Utilisateur de test pour TalkAI',
                    'country': 'France'
                })
                logger.info('✅ Utilisateur test créé')
            else:
                logger.info('ℹ️ Utilisateur test existe déjà')
            
            if admin_exists == 0:
                conn.execute(text("""
                    INSERT INTO users (email, hashed_password, nom, date_naissance, consent, role, bio, country)
                    VALUES (:email, :password, :nom, :date_naissance, :consent, :role, :bio, :country)
                """), {
                    'email': 'admin@talkai.com',
                    'password': admin_password_hash,
                    'nom': 'Admin TalkAI',
                    'date_naissance': '1985-03-20',
                    'consent': True,
                    'role': 'admin',
                    'bio': 'Administrateur principal de la plateforme TalkAI',
                    'country': 'France'
                })
                logger.info('✅ Utilisateur admin créé')
            else:
                logger.info('ℹ️ Utilisateur admin existe déjà')
            
            # La transaction est automatiquement commitée à la fin du bloc 'with'
        
        logger.info('🎉 Initialisation des utilisateurs terminée!')
        
    except Exception as e:
        logger.error(f'❌ Erreur création utilisateurs: {e}')
        raise e

if __name__ == "__main__":
    create_default_users()