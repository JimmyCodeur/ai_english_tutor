from fastapi import FastAPI, Depends, Form, HTTPException, status, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from bdd.schema_pydantic import UserCreate, User, CreateAdminRequest as PydanticUser
from bdd.schema_pydantic import TokenInBody, UserLoginResponse, TranslationRequest, ChatRequest
from pydantic import EmailStr, constr
from datetime import date
from bdd.database import get_db
from bdd.crud import create_user, delete_user, update_user, get_user_by_id, update_user_role
from bdd.models import Base, User as DBUser, Role
from token_security import get_current_user, authenticate_user, create_access_token, revoked_tokens
from log_conversation import log_conversation
from load_model import generate_phi3_response
from tts_utils import text_to_speech_audio
from transcribe_audio import transcribe_audio
from audio_utils import file_to_base64, process_audio_file
from detect_langs import detect_language
from cors import add_middleware
from prompt import prompt_tommy_start, prompt_tommy_fr, prompt_tommy_en, get_random_r_f_m_greetings_common_conversations, r_f_m_greetings_common_conversations, generate_response_variation, incorrect_responses_general, correct_responses_general, generate_correct_response, generate_incorrect_response
from TTS.api import TTS
import nltk
import soundfile as sf
import numpy as np
from fastapi import FastAPI, Query
from typing import Optional
from pydantic import BaseModel
from prompt import get_random_category 


app = FastAPI()
add_middleware(app)
app.mount("/static", StaticFiles(directory="./front/static"), name="static")
nltk.download('punkt')
current_prompt = None

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

@app.post("/chat/")
async def chat_with_brain(audio_file: UploadFile = File(...), voice: str = "english_ljspeech_tacotron2-DDC"):
    model_name = 'phi3'
    tts_model = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False, gpu=True)    
    tts_model.to("cuda")    
    audio_path = f"./audio/user/{audio_file.filename}"    
    with open(audio_path, "wb") as f:
        f.write(await audio_file.read())    
    denoised_audio_path = process_audio_file(audio_path, audio_file.filename) 
    # save_user_audio(denoised_audio_path)
    user_input = transcribe_audio(denoised_audio_path).strip()
    print(user_input)
    language = detect_language(user_input)
    print(language)       

    if user_input == "Thank you.":
        return {
            "user_input": user_input,
            "generated_response": None,
            "audio_base64": None 
        }
    
    log_file_path = './log/conversation_logs.txt'
    if user_input.lower() in ["help", "help.", "help!", "help?","Help !"]:
        return help_sugg(log_file_path, model_name, user_input)

    if language == 'unknown':
        return {
            "error": "unknown_language",
            "generated_response": "It appears you speak French. Please check your pronunciation.",
            "audio_base64": None,
        }
    
    if language == 'en':
        print("promp2 english tommy")
        prompt = prompt_tommy_en.format(user_input=user_input)
        print(prompt)
    elif language == 'fr':
        print("user parle francais")
        prompt = prompt_tommy_fr.format(user_input=user_input)
        print(prompt)

    generated_response = generate_phi3_response(prompt)
    audio_file_path = text_to_speech_audio(generated_response, voice)           
    audio_base64 = file_to_base64(audio_file_path)    
    log_conversation(prompt, generated_response)    
    return {
        "user_input": user_input,
        "generated_response": generated_response,
        "audio_base64": audio_base64
    }

class StartChatRequest(BaseModel):
    choice: Optional[str] = Query(None)

@app.post("/chat_repeat/start")
async def start_chat(request_data: StartChatRequest, voice: str = "english_ljspeech_tacotron2-DDC"):
    global current_prompt

    choice = request_data.choice
    
    if choice == "r_f_m_greetings_common_conversations":
        prompt = get_random_category("greetings_common_conversations")
    elif choice == "english_phrases":
        prompt = get_random_category("english_phrases")
    else:
        return {"error": "Invalid choice parameter."}

    if prompt:
        current_prompt = prompt.rstrip("!?.")
        print(prompt)
        
        # Si prompt est dans r_f_m_greetings_common_conversations, le retirer de la liste
        if choice == "r_f_m_greetings_common_conversations" and prompt in r_f_m_greetings_common_conversations:
            r_f_m_greetings_common_conversations.remove(prompt)
        
        generated_response = generate_response_variation(prompt)
        
        audio_file_path = text_to_speech_audio(generated_response, voice) 
        audio_base64 = file_to_base64(audio_file_path)    
        
        log_conversation(prompt, generated_response)    
        
        return {
            "generated_response": generated_response,
            "audio_base64": audio_base64
        }
    else:
        return {"error": "No more phrases available."}

@app.post("/chat_repeat")
async def chat_with_brain(audio_file: UploadFile = File(...), voice: str = "english_ljspeech_tacotron2-DDC"):
    global current_prompt
    print(current_prompt)

    audio_path = f"./audio/user/{audio_file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await audio_file.read())
    
    user_input = transcribe_audio(audio_path).strip().lower()   
    user_input = user_input.rstrip("!?.")
    
    print(f"User's voice input: {user_input}")
    expected_prompt = current_prompt.lower().strip()

    if current_prompt is None:
        generated_response = "There are no prompts available at the moment."
    else:
        expected_prompt = current_prompt.lower().strip()
    
        if user_input == expected_prompt or user_input == expected_prompt.rstrip("!?."):
            generated_response = generate_correct_response(correct_responses_general, current_prompt)
            
            if r_f_m_greetings_common_conversations:
                prompt = get_random_r_f_m_greetings_common_conversations()
                current_prompt = prompt
                
                r_f_m_greetings_common_conversations.remove(prompt)
                
                generated_response += f"\n\n{generate_response_variation(prompt)}"
            else:
                current_prompt = None
                generated_response += "\n\nThere are no more phrases available."
        else:
            generated_response = generate_incorrect_response(incorrect_responses_general, current_prompt)
    
        audio_file_path = text_to_speech_audio(generated_response, voice)
        audio_base64 = file_to_base64(audio_file_path)
        
        log_conversation(user_input, generated_response)
        
        return {
            "user_input": user_input,
            "generated_response": generated_response,
            "audio_base64": audio_base64
        }
        
@app.post("/translate")
async def translate_message(request: TranslationRequest):
    try:
        prompt = f"translate this message in french: \n {request.message}"
        generated_response = generate_phi3_response(prompt)

        return {
            "translated_message": generated_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/chat/message")
async def send_message(request: ChatRequest):
    try:
        prompt = request.message
        generated_response = generate_phi3_response(prompt)
        audio_file_path = text_to_speech_audio(generated_response, "english_ljspeech_tacotron2-DDC")
        audio_base64 = file_to_base64(audio_file_path)
        log_conversation(prompt, generated_response)
        return {
            "generated_response": generated_response,
            "audio_base64": audio_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_main:app", host="127.0.0.1", port=8000, reload=True)