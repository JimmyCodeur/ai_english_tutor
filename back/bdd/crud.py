from sqlalchemy.orm import Session
from models import User as DBUser
from schemas import UserCreate
from datetime import datetime

def create_user(db: Session, user: UserCreate):
    hashed_password = "randompassword"  # Remplacer par un vrai hachage de mot de passe
    db_user = DBUser(
        email=user.email,
        hashed_password=hashed_password,
        nom=user.nom,
        date_creation=datetime.utcnow()
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
