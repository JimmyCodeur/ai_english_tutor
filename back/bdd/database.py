from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from back.bdd.config import settings
import time
import sys
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# URL de la base de données
DATABASE_URL = settings.get_database_url()
logger.info(f"🔗 Connexion à la base de données: {DATABASE_URL.replace(settings.POSTGRES_PASSWORD, '***')}")

def create_db_engine():
    """Créer le moteur de base de données avec retry logic"""
    for attempt in range(10):
        try:
            engine = create_engine(
                DATABASE_URL,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=300,
                echo=False  # Mettre à True pour debug SQL
            )
            
            # Tester la connexion
            with engine.connect() as connection:
                result = connection.execute(text("SELECT version()"))
                version = result.fetchone()[0]
                logger.info(f"✅ Connexion PostgreSQL réussie: {version}")
            
            return engine
            
        except Exception as e:
            logger.error(f"❌ Échec connexion DB (tentative {attempt + 1}/10): {e}")
            if attempt < 9:
                logger.info("⏳ Nouvelle tentative dans 5 secondes...")
                time.sleep(5)
            else:
                logger.error("💥 Impossible de se connecter après 10 tentatives")
                sys.exit(1)

# Créer le moteur
engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Obtenir une session de base de données"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialiser la base de données"""
    from back.bdd.models import Base
    logger.info("🗃️ Initialisation des tables...")
    
    try:
        # Créer toutes les tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables créées/vérifiées avec succès")
        
        # Vérifier que les tables existent
        with engine.connect() as connection:
            result = connection.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            tables = [row[0] for row in result.fetchall()]
            logger.info(f"📋 Tables disponibles: {', '.join(tables)}")
            
    except Exception as e:
        logger.error(f"❌ Erreur initialisation DB: {e}")
        raise

def check_db_health():
    """Vérifier la santé de la base de données"""
    try:
        with engine.connect() as connection:
            # Test de base
            connection.execute(text("SELECT 1"))
            
            # Vérifier les utilisateurs
            result = connection.execute(text("SELECT COUNT(*) FROM users"))
            user_count = result.fetchone()[0]
            
            logger.info(f"💓 DB Health OK - {user_count} utilisateurs")
            return True
            
    except Exception as e:
        logger.error(f"💔 DB Health FAIL: {e}")
        return False