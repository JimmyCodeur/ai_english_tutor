from sqlalchemy import Column, Integer, String, Date, Enum as SqlEnum, func, Text
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


