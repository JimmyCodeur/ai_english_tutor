#fichier schema_pydantic.py
from pydantic import BaseModel, EmailStr
from datetime import datetime, date
from pydantic import BaseModel
from typing import Optional, List, Tuple
from fastapi import Query

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
    consent: bool

class User(UserBase):
    id: int
    date_creation: datetime
    consent: bool
    role: str 
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    country: Optional[str] = None
    
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
    voice: Optional[str] = "bark"

class ConversationLog(BaseModel):
    id: int
    user_id: int
    timestamp: datetime
    prompt: str
    response: str
    user_audio_base64: Optional[str] = None
    user_input: Optional[str] = None

    class Config:
        orm_mode = True

class ConversationLogCreate(ConversationLog):
    pass

class ConversationLog(ConversationLog):
    id: int
    
class ConversationSchema(BaseModel):
    id: int
    user1_id: int
    user2_id: int
    category: str
    start_time: datetime

    class Config:
        orm_mode = True

class MessageSchema(BaseModel):
    id: int
    user_id: int
    conversation_id: int
    timestamp: datetime
    content: str
    user_input: Optional[str]
    user_audio_base64: Optional[str]
    ia_audio_base64: Optional[str]
    response: Optional[str]
    marker: Optional[str]
    suggestion: Optional[str]

    class Config:
        orm_mode = True

class StartChatRequest(BaseModel):
    choice: Optional[str] = Query(None)

class UnclearResponse(BaseModel):
    response: str
    suggestion: str

class AnalysisResponse(BaseModel):
    translations: List[Tuple[str, str]]
    unclear_responses: List[UnclearResponse]
    french_responses: List[str]
    french_responses_count: int
    unclear_responses_count: int
    translations_count: int
    correct_phrases_count: int
    duration: str
    category: str
    total_messages: int
    avg_user_response_time: str

class SuggestionRequest(BaseModel):
    unclear_response: str


