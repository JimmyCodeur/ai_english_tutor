#!/usr/bin/env python3
import sys
sys.path.append('/app')

from back.bdd.database import engine
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_enum():
    """Corriger l'enum PostgreSQL"""
    try:
        with engine.begin() as conn:
            # Supprimer l'enum existant et le recréer
            conn.execute(text("DROP TYPE IF EXISTS role_enum CASCADE"))
            conn.execute(text("CREATE TYPE role_enum AS ENUM ('utilisateur', 'admin')"))
            
            # Recréer la colonne avec le bon enum
            conn.execute(text("ALTER TABLE users ALTER COLUMN role TYPE role_enum USING role::text::role_enum"))
            
            logger.info('✅ Enum corrigé')
            
    except Exception as e:
        logger.error(f'❌ Erreur: {e}')
        raise e

if __name__ == "__main__":
    fix_enum()