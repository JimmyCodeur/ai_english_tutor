from pydantic import BaseSettings
import os

class Settings(BaseSettings):
    # Configuration PostgreSQL
    POSTGRES_USER: str = "talkai_user"
    POSTGRES_PASSWORD: str = "talkai_password"
    POSTGRES_DB: str = "talkai_db"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"
    
    # Configuration JWT
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 heures
    
    # Configuration Ollama
    OLLAMA_HOST: str = "http://localhost:11434"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def get_database_url(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

settings = Settings()

# Afficher la configuration (sans les mots de passe)
print(f"🔗 Database Host: {settings.POSTGRES_HOST}")
print(f"🔗 Database Name: {settings.POSTGRES_DB}")
print(f"🔗 Database User: {settings.POSTGRES_USER}")
print(f"🤖 Ollama Host: {settings.OLLAMA_HOST}")