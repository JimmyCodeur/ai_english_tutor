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
    date_naissance: date
    password: str

class UserLoginResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    email: EmailStr
    nom: str

class MessageRequest(BaseModel):
    message: str

class SuggestionResponse(BaseModel):
    suggestions: list

class TranslationRequest(BaseModel):
    message: str  

class ChatRequest(BaseModel):
    message: str

# class ChatStartRequest(BaseModel):
#     category: str
#     voice: str