from pydantic import BaseModel
from datetime import datetime, date
from typing import Optional

class UserBase(BaseModel):
    email: str
    nom: Optional[str] = None
    date_naissance: Optional[date] = None

class UserCreate(UserBase):
    hashed_password: str

class User(UserBase):
    id: int
    date_creation: datetime

    class Config:
        from_attributes = True

class SessionBase(BaseModel):
    user_id: int
    start_time: datetime
    end_time: datetime

class SessionCreate(SessionBase):
    pass

class Session(SessionBase):
    id: int

    class Config:
        from_attributes = True
