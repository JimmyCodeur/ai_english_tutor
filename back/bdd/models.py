from sqlalchemy import Column, Integer, String, Date, Enum as SqlEnum, func, DateTime, ForeignKey
from sqlalchemy import Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from datetime import datetime, timezone

Base = declarative_base()

class Role(str, PyEnum):
    UTILISATEUR = "utilisateur"
    ADMIN = "admin"

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    nom = Column(String, nullable=True)
    date_naissance = Column(Date, nullable=True)
    consent = Column(Boolean, nullable=False)
    role = Column(SqlEnum(Role), default=Role.UTILISATEUR, nullable=False) 
    avatar = Column(String, nullable=True)


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