from fastapi import FastAPI, Depends, Form, HTTPException, status, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from bdd.schema_pydantic import UserCreate, User, CreateAdminRequest as PydanticUser
from bdd.schema_pydantic import TokenInBody, UserLoginResponse
from pydantic import EmailStr, constr
from datetime import date
from bdd.database import get_db, engine
from bdd.crud import create_user, delete_user, update_user, get_user_by_id, update_user_role
from bdd.models import Base, User as DBUser, Role
from passlib.context import CryptContext
from token_security import get_current_user, authenticate_user, create_access_token, revoked_tokens
from log_conversation import log_conversation
from load_model import generate_ollama_response
from tts_utils import text_to_speech_audio
from app import transcribe_audio
from TTS.api import TTS
from fastapi.middleware.cors import CORSMiddleware
import base64
import nltk
from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException


DetectorFactory.seed = 0  # Pour obtenir des résultats reproductibles

def detect_language(text):
    try:
        languages = detect_langs(text)
        # Filtrer pour ne garder que les langues en et fr
        filtered_langs = [lang for lang in languages if lang.lang in ['en', 'fr']]
        if not filtered_langs:
            return 'unknown'
        # Sélectionner la langue avec le score de confiance le plus élevé
        best_guess = max(filtered_langs, key=lambda lang: lang.prob)
        # Utiliser un seuil de confiance
        if best_guess.prob < 0.7:  # Ajuster le seuil en fonction des tests
            return 'unknown'
        return best_guess.lang
    except LangDetectException:
        return 'unknown'

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise tous les domaines (à ajuster en fonction de vos besoins)
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="./front/static"), name="static")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get('/')
def index():
    return FileResponse('./front/form-login.html')

@app.get("/users-profile.html")
def get_users_profile():
    return FileResponse('./front/users-profile.html')

@app.get("/form-register.html")
def get_users_profile():
    return FileResponse('./front/form-register.html')

@app.get("/form-login.html")
def get_users_profile():
    return FileResponse('./front/form-login.html')

@app.get("/profile-setting.html")
def get_users_profile():
    return FileResponse('./front/profile-setting.html')

@app.get("/brain-info-course.html")
def get_users_profile():
    return FileResponse('./front/brain-info-course.html')

@app.get("/brain-course.html")
def get_users_profile():
    return FileResponse('./front/brain-course.html')

@app.get("/home.html")
def get_users_profile():
    return FileResponse('./front/home.html')

@app.get("/users/me", response_model=User)
def read_current_user(current_user: User = Depends(get_current_user)):
    return current_user

@app.get("/users/{user_id}", response_model=PydanticUser)
def read_user_by_id(user_id: int, db: Session = Depends(get_db)):
    user = get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/login/", response_model=UserLoginResponse)
def login_for_access_token(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = authenticate_user(db, email, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email or password incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user.email, "role": user.role})
    print(f"Generated Access Token: {access_token}")
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.id,
        "email": user.email,
        "nom": user.nom
    }


@app.post("/logout")
def logout(token_data: TokenInBody, db: Session = Depends(get_db)):
    token = token_data.token
    if token is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Token missing")
    
    if token in revoked_tokens:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token already revoked")

    revoked_tokens.add(token)
    return {"message": "Successfully logged out"}


@app.post("/users/", response_model=User)
def create_user_endpoint(
    email: EmailStr = Form(...),
    nom: str = Form(...),
    date_naissance: date = Form(..., description="Format attendu : YYYY-MM-DD"),
    password: constr(min_length=6) = Form(..., description="Mot de passe de l'utilisateur (minimum 6 caractères)"),
    db: Session = Depends(get_db)
):
    user_data = UserCreate(
        email=email,
        nom=nom,
        date_naissance=date_naissance.isoformat(),
        password=password
    )
    try:
        db_user = create_user(db=db, user=user_data)
        return db_user
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))

@app.post("/token", response_model=dict)
def login_for_access_token(
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = authenticate_user(db, email, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email or password incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.email, "role": user.role})
    return {"access_token": access_token, "token_type": "bearer"}

@app.put("/update-user-role/{user_id}")
def update_user_role_endpoint(
    user_id: int,
    new_role: str,
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_user)
):
    if current_user.role != Role.ADMIN.name:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Insufficient permissions")
    
    updated = update_user_role(db=db, user_id=user_id, new_role=new_role)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Utilisateur non trouvé")
    
    return {"message": f"Rôle de l'utilisateur {user_id} mis à jour avec succès"}

@app.put("/users/{user_id}")
def update_user_route(
    user_id: int,
    email: EmailStr = Form(None),
    nom: str = Form(None),
    date_naissance: date = Form(None),
    password: str = Form(None),
    db: Session = Depends(get_db)
):
    user_data = UserCreate(
        email=email,
        nom=nom,
        date_naissance=date_naissance,
        password=password
    )
    updated_user = update_user(db=db, user_id=user_id, user_data=user_data)
    return updated_user

@app.delete("/users/{user_id}")
def delete_user_route(user_id: int, db: Session = Depends(get_db)):
    deleted = delete_user(db=db, user_id=user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    return {"message": "Utilisateur supprimé avec succès"}

nltk.download('punkt')
first_interaction = True

@app.post("/chat/")
async def chat_with_brain(audio_file: UploadFile = File(...)):
    global first_interaction

    model_name = 'phi3'

    tts_model = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False, gpu=True)
    
    tts_model.to("cuda")
    
    audio_path = f"user_audio/{audio_file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await audio_file.read()) 
    
    user_input = transcribe_audio(audio_path).strip()
    print(user_input)

    if not user_input:
        raise HTTPException(status_code=400, detail="Aucune entrée utilisateur détectée dans l'audio.")
    
    language = detect_language(user_input)
    print(language)

    if language == 'unknown':
        return {
            "user_input": user_input,
            "generated_response": False,
            "audio_base64": None,
            "display_message": True
        }
    
    if language == 'en':
        prompt = (
            "You are a 4-year-old child named Tommy. You have to converse with a French person who doesn't speak English. \n"
            "The idea is that you always have to ask questions, find conversation topics, and help them learn English through immersion.\n"
            "1. Engage the person in simple and playful conversations in English to encourage learning.\n"
            "2. Ask about their day or their interests to keep the conversation going.\n"
            "3. Use simple English words to explain things they might not understand.\n"
            "4. Do not ask the same question more than once.\n"
            "Remember, the goal is to create a fun and engaging environment where they can learn English naturally.\n"
            "Please limit greetings (e.g., Hi, hello, hi there) to only once in your response.\n"
            "Please limit the number of questions asked to one per response.\n"
            "Avoid asking how to say things in French.\n"
            f"User: {user_input}\n"
        )
    else:
        prompt = (
            "You are a 4-year-old child named Tommy. \n"
            "Whenever the user speaks in French, you should respond that you don't understand and ask them to speak in English. \n "
            "You must always respond in English.\n"
            "Warning: Poorly pronounced English detected. Please try again with clearer pronunciation."
            f"User: {user_input}\n"
        )

    generated_response = generate_ollama_response(model_name, prompt)
    
    if isinstance(generated_response, list):
        generated_response = ' '.join(generated_response)
    
    sentences = nltk.tokenize.sent_tokenize(generated_response)
    
    if language == 'en':
        limited_response = ' '.join(sentences[:3])
    else:
        limited_response = ' '.join(sentences[:2])
    
    generated_response = limited_response
    
    audio_file_path = text_to_speech_audio(generated_response)
           
    audio_base64 = file_to_base64(audio_file_path)
    
    log_conversation(prompt, generated_response)
    
    return {
        "user_input": user_input,
        "generated_response": generated_response,
        "audio_base64": audio_base64
    }

@app.post("/chat/start")
async def start_chat():
    model_name = 'phi3'
    prompt = (
            "Hello! My name is Tommy. I'm a 4-year-old child. Let's have a fun chat in English! "
            "The idea is that you always have to ask questions, find conversation topics, and help them learn English through immersion.\n"
            "1. Engage the person in simple and playful conversations in English to encourage learning.\n"
            "2. Ask about their day or their interests to keep the conversation going.\n"
            "3. Use simple English words to explain things they might not understand.\n"
            "4. Do not ask the same question more than once.\n"
            "Remember, the goal is to create a fun and engaging environment where they can learn English naturally.\n"
            "Please limit greetings (e.g., Hi, hello, hi there) to only once in your response.\n"
            "Please limit the number of questions asked to one per response.\n"
            "Avoid asking how to say things in French.\n"
    )

    generated_response = generate_ollama_response(model_name, prompt)

    if isinstance(generated_response, list):
        generated_response = ' '.join(generated_response)
    
    sentences = nltk.tokenize.sent_tokenize(generated_response)
    
    limited_response = ' '.join(sentences[:3])

    generated_response = limited_response
    
    audio_file_path = text_to_speech_audio(generated_response)
           
    audio_base64 = file_to_base64(audio_file_path)
    
    log_conversation(prompt, generated_response)
    
    return {
        "generated_response": generated_response,
        "audio_base64": audio_base64
    }

def file_to_base64(file_path):
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode("utf-8")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_main:app", host="127.0.0.1", port=8000, reload=True)
