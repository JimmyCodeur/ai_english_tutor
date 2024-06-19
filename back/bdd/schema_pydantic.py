from pydantic import BaseModel, EmailStr
from datetime import datetime, date
from pydantic import BaseModel
from typing import Optional

class TokenData(BaseModel):
    user_id: Optional[str] = None

class TokenInBody(BaseModel):
    token: str

class UserBase(BaseModel):
    email: EmailStr
    nom: str
    date_naissance: date

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    date_creation: datetime

    class Config:
        orm_mode = True

class CreateAdminRequest(BaseModel):
    email: str
    nom: str
    date_naissance: str
    password: str
