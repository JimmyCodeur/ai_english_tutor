from sqlalchemy import Column, Integer, String, Date, Enum as SqlEnum, func, DateTime
from sqlalchemy.ext.declarative import declarative_base
from enum import Enum as PyEnum
from datetime import date

Base = declarative_base()

class Role(str, PyEnum):
    UTILISATEUR = "utilisateur"
    ADMIN = "admin"

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String)
    nom = Column(String)
    date_naissance = Column(Date)
    date_creation = Column(Date, server_default=func.current_date())
    role = Column(SqlEnum(Role), default=Role.UTILISATEUR.name, nullable=False)
    avatar = Column(String, nullable=True)

class ConversationLog(Base):
    __tablename__ = 'conversation_logs'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)  # L'id de l'utilisateur qui a initié la conversation
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    prompt = Column(String, nullable=False)    # Le message d'entrée (prompt)
    response = Column(String)                  # La réponse générée par l'IA

    def __repr__(self):
        return f"<ConversationLog(id={self.id}, user_id={self.user_id}, timestamp={self.timestamp}, prompt='{self.prompt}', response='{self.response}')>"
