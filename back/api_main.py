from fastapi import FastAPI, Depends, Form, HTTPException, status, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from bdd.schema_pydantic import UserCreate, User, CreateAdminRequest as PydanticUser
from bdd.schema_pydantic import TokenInBody, UserLoginResponse
from pydantic import EmailStr, constr
from datetime import date
from bdd.database import get_db
from bdd.crud import create_user, delete_user, update_user, get_user_by_id, update_user_role
from bdd.models import Base, User as DBUser, Role
from token_security import get_current_user, authenticate_user, create_access_token, revoked_tokens
from log_conversation import log_conversation
from load_model import generate_ollama_response, generate_phi3_response
from tts_utils import text_to_speech_audio
from transcribe_audio import transcribe_audio
from audio_utils import file_to_base64,save_user_audio
from detect_langs import detect_language
from cors import add_middleware
from prompt import prompt_tommy_start
from TTS.api import TTS
import base64
import nltk
import noisereduce
import soundfile as sf
import numpy as np
import subprocess
import os

app = FastAPI()
add_middleware(app)

app.mount("/static", StaticFiles(directory="./front/static"), name="static")

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

@app.post("/chat/")
async def chat_with_brain(audio_file: UploadFile = File(...)):

    model_name = 'phi3'

    tts_model = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False, gpu=True)    
    tts_model.to("cuda")
    
    audio_path = f"./user_audio/{audio_file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await audio_file.read()) 

    converted_audio_path = f"./audio/user/converted_{audio_file.filename}"
    if os.path.exists(converted_audio_path):
        os.remove(converted_audio_path)
    subprocess.run(['ffmpeg', '-i', audio_path, '-acodec', 'pcm_s16le', '-ar', '44100', converted_audio_path])

    audio_data, sr = sf.read(converted_audio_path)
    reduced_noise = noisereduce.reduce_noise(audio_data, sr)

    denoised_audio_path = f"./audio/user/denoised_converted_{audio_file.filename}"
    sf.write(denoised_audio_path, reduced_noise, sr)

    # audio_path = save_user_audio(audio_file)

    user_input = transcribe_audio(audio_path).strip()
    print(user_input)

    language = detect_language(user_input)
    print(language)       

    if user_input == "Thank you.":
        return {
            "user_input": user_input,
            "generated_response": None,
            "audio_base64": None 
        }
    
    if user_input.lower() in ["help", "help.", "help!", "help?"]:
        print("Detected help request.")
        
        with open('./log/conversation_logs.txt', 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]
            last_line = lines[-1] if lines else ""

        if last_line.startswith('[') and 'Response: ' in last_line:
            prompt_for_suggestions = last_line.split('Response: ', 1)[1].strip()
        else:
            prompt_for_suggestions = last_line

        prompt_for_suggestions += (
        "\n Please respond with very short messages and at an easy level"        
        )

        print(prompt_for_suggestions)

        generated_suggestions = generate_ollama_response(model_name, prompt_for_suggestions)
        
        if isinstance(generated_suggestions, list):
            generated_suggestions = '\n'.join([f"{i + 1}. {s}" for i, s in enumerate(generated_suggestions)])

        suggestions_list = generated_suggestions.split('\n')[:2]
        print(suggestions_list)

        return {
            "user_input": user_input,
            "generated_response": None,
            "audio_base64": None,
            "suggestions": suggestions_list
        }

    if not user_input:
        raise HTTPException(status_code=400, detail="Aucune entrée utilisateur détectée dans l'audio.")

    if language == 'unknown':
        return {
            "error": "unknown_language",
            "generated_response": "What ?",
            "audio_base64": None,
        }
    
    if language == 'en':
        print("promp2 english tommy")
        prompt = (
            "You are a 4-year-old child named Tommy. You have to converse with a French person who doesn't speak English. \n"
            "The idea is that you always have to ask questions, find conversation topics, and help them learn English through immersion.\n"
            "1. Engage the person in simple and playful conversations in English to encourage learning.\n"
            "2. Ask about their day or their interests to keep the conversation going.\n"
            "3. Use simple English words to explain things they might not understand.\n"
            "4. Do not ask the same question more than once.\n"
            "5. You must only repeat the same phrase once"
            "Remember, the goal is to create a fun and engaging environment where they can learn English naturally.\n"
            "Please limit greetings (e.g., Hi, hello, Hello, hi there) to only once in your response.\n"
            "Please limit the number of questions asked to one per response.\n"
            "Avoid asking how to say things in French.\n"
            f"User: {user_input}\n"
        )
    elif language == 'fr':
        print("user parle francais")
        prompt = (
            "You are a 4-year-old child named Tommy. \n"
            "Whenever the user speaks in French, you should respond that you don't understand and ask them to speak in English. \n "
            "You must always respond in English.\n"
            "Warning: Poorly pronounced English detected. Please try again with clearer pronunciation."
            f"User: {user_input}\n"
        )

    generated_response = generate_phi3_response(prompt)        
   
    # if language == 'en':
    #     limited_response = ' '.join(sentences[:3])
    # else:
    #     limited_response = ' '.join(sentences[:2])
       
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
    print("prompt_tommy_start")
    prompt = prompt_tommy_start
    generated_response = generate_phi3_response(prompt)    
    audio_file_path = text_to_speech_audio(generated_response)
    audio_base64 = file_to_base64(audio_file_path)    
    log_conversation(prompt, generated_response)    
    return {
        "generated_response": generated_response,
        "audio_base64": audio_base64
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_main:app", host="127.0.0.1", port=8000, reload=True)
