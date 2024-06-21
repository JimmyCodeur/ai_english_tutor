from sqlalchemy.orm import Session
from bdd.schema_pydantic import UserCreate, User
from datetime import datetime
from passlib.context import CryptContext
from sqlalchemy.exc import IntegrityError
from pydantic import ValidationError
from fastapi import HTTPException
from bdd.models import User as DBUser, Role
from datetime import date

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

def user_exists(db: Session, email: str) -> bool:
    return db.query(DBUser).filter(DBUser.email == email).first() is not None

def get_user_by_id(db: Session, user_id: int):
    return db.query(DBUser).filter(DBUser.id == user_id).first()

def get_all_users(db: Session):
    return db.query(DBUser).all()

def validate_password_length(password: str):
    if len(password) < 6:
        raise ValueError("Le mot de passe doit avoir au moins 6 caractères")
    
def get_password_hash(password):
    return pwd_context.hash(password)

def user_exists(db: Session, email: str) -> bool:
    return db.query(DBUser).filter(DBUser.email == email).first() is not None

def create_user(db: Session, user: UserCreate) -> User:
    if user_exists(db, user.email):
        raise ValueError("Email déjà enregistré")

    hashed_password = get_password_hash(user.password)
    db_user = DBUser(
        email=user.email,
        hashed_password=hashed_password,
        nom=user.nom,
        date_naissance=user.date_naissance,
        date_creation=datetime.utcnow()
    )
    
    try:
        db.add(db_user)
        db.commit()
        db.refresh(db_user)

        return User(
            id=db_user.id,
            email=db_user.email,
            nom=db_user.nom,
            date_naissance=db_user.date_naissance,
            date_creation=db_user.date_creation
        )
    
    except IntegrityError as e:
        db.rollback()
        raise ValueError("Erreur d'intégrité de la base de données") from e
    
def update_user_role(db: Session, user_id: int, new_role: str):
    user = db.query(DBUser).filter(DBUser.id == user_id).first()
    if not user:
        return False
    
    user.role = new_role
    db.commit()
    db.refresh(user)
    return True

def delete_user(db: Session, user_id: int):
    user = db.query(DBUser).filter(DBUser.id == user_id).first()
    if user:
        db.delete(user)
        db.commit()
        return True
    return False

def update_user(db: Session, user_id: int, user_data: UserCreate):
    user = db.query(DBUser).filter(DBUser.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    
    if user_data.email:
        user.email = user_data.email
    
    if user_data.nom:
        user.nom = user_data.nom
    
    if user_data.date_naissance:
        user.date_naissance = user_data.date_naissance
    
    if user_data.password:
        hashed_password = get_password_hash(user_data.password)
        user.hashed_password = hashed_password
    
    db.commit()
    db.refresh(user)
    return user
