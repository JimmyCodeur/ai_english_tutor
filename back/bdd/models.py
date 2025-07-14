from sqlalchemy import Column, Integer, String, Date, Enum as SqlEnum, func, DateTime, ForeignKey, Boolean, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from datetime import datetime, timezone

Base = declarative_base()

class Role(str, PyEnum):
    UTILISATEUR = "utilisateur"  # ← La valeur DOIT être en minuscules
    ADMIN = "admin"              # ← La valeur DOIT être en minuscules

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    nom = Column(String(255), nullable=True)
    date_naissance = Column(Date, nullable=True)
    consent = Column(Boolean, nullable=False, default=False)
    role = Column(SqlEnum(Role, values_callable=lambda obj: [e.value for e in obj]), default=Role.UTILISATEUR, nullable=False)
    date_creation = Column(DateTime, default=func.now(), nullable=False)
    
    # Nouveaux champs
    avatar_url = Column(String(500), nullable=True)
    bio = Column(Text, nullable=True)
    country = Column(String(100), nullable=True)

class ConversationLog(Base):
    __tablename__ = 'conversation_logs'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    prompt = Column(String, nullable=False) 
    response = Column(String)
    user_audio_base64 = Column(String)
    user_input = Column(String) 
    
    def __repr__(self):
        return f"<ConversationLog(id={self.id}, user_id={self.user_id}, timestamp={self.timestamp}, prompt='{self.prompt}', response='{self.response}')>"
    
class Conversation(Base):
    __tablename__ = 'conversations'
    id = Column(Integer, primary_key=True, index=True)
    user1_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    user2_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    category = Column(String, nullable=False)
    active = Column(Boolean, default=True)
    start_time = Column(DateTime, default=func.now(), nullable=False)
    end_time = Column(DateTime, nullable=True)
    user1 = relationship("User", foreign_keys=[user1_id])
    user2 = relationship("User", foreign_keys=[user2_id])
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, user1_id={self.user1_id}, user2_id={self.user2_id}, category='{self.category}')>"
    
    def end_session(self, db_session):
        if self.active:
            self.active = False
            self.end_time = datetime.utcnow().replace(tzinfo=timezone.utc)
            db_session.commit()

class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    conversation_id = Column(Integer, ForeignKey('conversations.id'), nullable=False)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    content = Column(String, nullable=False)
    user_input = Column(String, nullable=True)
    user_audio_base64 = Column(String, nullable=True)
    ia_audio_base64 = Column(String, nullable=True)
    response = Column(String, nullable=True)
    marker = Column(String, nullable=True)
    suggestion = Column(String, nullable=True)
    ia_audio_duration = Column(Float, nullable=True)
    user = relationship("User")
    conversation = relationship("Conversation")
    
    def __repr__(self):
        return f"<Message(id={self.id}, user_id={self.user_id}, conversation_id={self.conversation_id}, timestamp={self.timestamp}, content='{self.content}')>"