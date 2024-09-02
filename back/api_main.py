from fastapi import FastAPI, Depends, Form, HTTPException, status, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import asc
from back.bdd.schema_pydantic import UserCreate, User, CreateAdminRequest as PydanticUser
from back.bdd.schema_pydantic import TokenInBody, UserLoginResponse, TranslationRequest, ChatRequest, ConversationSchema, MessageSchema, StartChatRequest, AnalysisResponse
from back.bdd.crud import create_user, delete_user, update_user, get_user_by_id, update_user_role
from back.bdd.models import Message, User as DBUser, Role
from back.bdd.models import Conversation
from back.bdd.log_conversation import create_or_get_conversation, log_conversation_to_db, log_message_to_db, log_conversation_and_message
from back.bdd.database import get_db
from back.prompt.alice_conv import generate_ai_response_alice 
from back.prompt.prompt import generate_response_variation, get_prompt, get_category, handle_response, category_mapping
from pydantic import EmailStr, constr, BaseModel
from back.token_security import get_current_user, authenticate_user, create_access_token, revoked_tokens
from back.load_model import generate_phi3_response, generate_ollama_response
from back.help_suggestion import help_sugg
from back.tts_utils import text_to_speech_audio
from back.transcribe_audio import transcribe_audio
from back.audio_utils import file_to_base64, is_valid_audio_file, delete_audio_file
from back.detect_langs import detect_language
from back.cors import add_middleware
from typing import List ,Optional
from datetime import timezone, timedelta, date, datetime
import nltk
import time
import re
import spacy
import aiofiles
import asyncio

app = FastAPI()
add_middleware(app)
app.mount("/static", StaticFiles(directory="./front/static"), name="static")
nltk.download('punkt')
current_prompt = None
nlp = spacy.load("en_core_web_sm")
conversation_history = {}
conversation_start_time = {}

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

@app.get("/course.html")
def get_users_profile():
    return FileResponse('./front/course.html')

@app.get("/home.html")
def get_users_profile():
    return FileResponse('./front/home.html')

@app.get("/conversation.html")
def get_users_profile():
    return FileResponse('./front/conversation.html')

@app.get("/analysis.html")
def get_users_profile():
    return FileResponse('./front/analysis.html')

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
    # consent: bool = Form(..., description="Consentement de l'utilisateur"),
    db: Session = Depends(get_db)
):
    print(f"Reçu : email={email}, nom={nom}, date_naissance={date_naissance}, password={password}, consent={consent}")
    
    user_data = UserCreate(
        email=email,
        nom=nom,
        date_naissance=date_naissance,
        password=password,
        # consent=consent
    )
    try:
        db_user = create_user(db=db, user=user_data)
        return db_user
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

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
    print(f"Generated Access Token: {access_token}")
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

def calculate_avg_user_response_time(messages):
    if not messages:
        return "N/A"

    messages.sort(key=lambda msg: msg.timestamp)    
    time_differences = []
    for i in range(1, len(messages)):
        if messages[i].user_id == messages[i-1].user_id: 
            time_diff = (messages[i].timestamp - messages[i-1].timestamp).total_seconds()
            if messages[i-1].ia_audio_duration:
                time_diff -= messages[i-1].ia_audio_duration
            time_differences.append(time_diff)

    if not time_differences:
        return "N/A"
    
    avg_response_time = sum(time_differences) / len(time_differences)
    avg_response_time_delta = timedelta(seconds=avg_response_time)
    return str(avg_response_time_delta).split('.')[0]

@app.get("/analyze_session/{conversation_id}", response_model=AnalysisResponse)
def analyze_session(conversation_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    translations = db.query(Message).filter(
        Message.conversation_id == conversation_id,
        Message.marker == "translations"
    ).all()

    unclear_responses = db.query(Message).filter(
        Message.conversation_id == conversation_id,
        Message.marker == "unclear_responses"
    ).all()

    french_responses = db.query(Message).filter(
        Message.conversation_id == conversation_id,
        Message.marker == "french_responses"
    ).all()

    all_messages = db.query(Message).filter(Message.conversation_id == conversation_id).all()
    correct_phrases_count = len([msg for msg in all_messages if msg.marker not in ["unclear_responses", "french_responses"]])

    if conversation.end_time:
        duration = conversation.end_time - conversation.start_time
    else:
        duration = datetime.utcnow() - conversation.start_time
    
    duration_str = str(duration).split('.')[0]
    total_messages = len(all_messages)
    avg_user_response_time = calculate_avg_user_response_time(all_messages)

    analysis_result = {
        "translations": [(msg.user_input, msg.response) for msg in translations],
        "unclear_responses": [{"response": msg.content, "suggestion": msg.suggestion} for msg in unclear_responses],
        "french_responses": [msg.content for msg in french_responses],
        "french_responses_count": len(french_responses),
        "unclear_responses_count": len(unclear_responses),
        "translations_count": len(translations),
        "correct_phrases_count": correct_phrases_count,
        "duration": duration_str,
        "category": conversation.category,
        "total_messages": total_messages,
        "avg_user_response_time": avg_user_response_time
    }
    
    return analysis_result

@app.post("/get_suggestions")
def get_suggestions(request: dict, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    unclear_response = request.get('unclear_response', '')

    if unclear_response:
        prompt = f"Improve the following unclear response in one sentence easy: {unclear_response}"
        suggestion = generate_phi3_response(prompt)
        return {"suggestion": suggestion}
    else:
        raise HTTPException(status_code=400, detail="Unclear response not provided.")

@app.post("/chat_conv/start")
async def start_chat_with_brain(request_data: StartChatRequest, voice: str = "english_ljspeech_tacotron2-DDC", db: Session = Depends(get_db), current_user: int = Depends(get_current_user)):
    global conversation_history
    global conversation_start_time

    user_id = current_user.id
    choice = request_data.choice

    if user_id not in conversation_history:
        user_history = []
        conversation_history[user_id] = user_history
        conversation_start_time[user_id] = time.time()
    else:
        user_history = conversation_history[user_id]

    alice_start_greetings = "Hello! I'm really happy to meet you. How are you?"
    user_history.append({'input': "", 'response': alice_start_greetings})

    audio_file_path, duration = await text_to_speech_audio(alice_start_greetings, voice)
    audio_base64 = file_to_base64(audio_file_path)
    delete_audio_file(audio_file_path)

    log_conversation_to_db(db, user_id, alice_start_greetings, alice_start_greetings)
    conversation = create_or_get_conversation(db, user_id, choice)
    log_message_to_db(db, user_id, conversation.id, alice_start_greetings, alice_start_greetings, audio_base64, ia_audio_duration=duration)

    return {
        "generated_response": alice_start_greetings,
        "audio_base64": audio_base64,
        "conversation_history": user_history
    }

def evaluate_sentence_quality_phi3(sentence: str) -> bool:
    print("Début sentence")
    prompt = f"Is the following sentence understandable and acceptable for simple conversation, ignoring small grammar mistakes, capitalization, and punctuation? Answer 'yes' or 'no' only:\n\n{sentence}"
    print(prompt)
    
    model_name = 'phi3'
    response = generate_ollama_response(model_name, prompt).strip().lower()
    response_short = response.split(".")[0]
    print(f"Response from model: {response_short}")
    
    if "capitalization" in response or "punctuation" in response or "comma" in response:
        print("Ignoring minor errors in capitalization or punctuation")
        return True

    if "yes" in response_short or "understandable" in response:
        return True
    
    return response_short.startswith("yes")

def detect_translation_request(user_input: str) -> Optional[str]:
    match = re.search(r'how (do|did|can|could|would|will|should) (you|i|we) say[,]? in english[,]?\s*(.*)', user_input, re.IGNORECASE)
    if match and match.group(3):
        return match.group(3).strip()
    return None

@app.post("/chat_conv")
async def chat_with_brain(
    choice: str = Form(...),
    audio_file: UploadFile = File(...),
    voice: str = "english_ljspeech_tacotron2-DDC",
    db: Session = Depends(get_db),
    current_user: int = Depends(get_current_user)
):
    global conversation_history
    global conversation_start_time

    user_id = current_user.id
    category = get_category(choice)  

    if not is_valid_audio_file(audio_file):
        raise HTTPException(status_code=400, detail="Invalid audio file format")

    user_audio_path = f"./audio/user/{audio_file.filename}"
    async with aiofiles.open(user_audio_path, "wb") as f:
        await f.write(await audio_file.read())
    print("Début transcription audio")
    transcription_task = asyncio.create_task(transcribe_audio(user_audio_path))
    user_audio_base64 = await asyncio.to_thread(file_to_base64, user_audio_path)    

    user_input, transcription_time = await transcription_task
    print(f"Transcription terminée en secondes")
    user_input = user_input.strip().lower()
    phrase_to_translate = detect_translation_request(user_input)
    print(f"detect terminée en secondes")
    
    if phrase_to_translate:
        prompt = f"Translate this sentence into English without adding comments: {phrase_to_translate}"
        translated_phrase = await asyncio.to_thread(generate_phi3_response, prompt)
        generated_response = f"{translated_phrase}"      
        audio_file_path, duration = await text_to_speech_audio(generated_response, voice)
        audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path) 
        await asyncio.to_thread(delete_audio_file, audio_file_path)
        log_conversation_and_message(db, user_id, category, user_input, user_input, generated_response, user_audio_base64, audio_base64, marker="translations", ia_audio_duration=duration)
        
        return {
            "user_input": user_input,
            "generated_response": generated_response,
            "audio_base64": audio_base64,
            "conversation_history": None,
            "user_audio_base64": user_audio_base64,
            "metrics": {
                "transcription_time": transcription_time,
            }
        }

    if user_id not in conversation_history:
        conversation_history[user_id] = []
        conversation_start_time[user_id] = time.time()

    user_history = conversation_history[user_id]    
    language = detect_language(user_input)    
    current_time = time.time()
    start_time = conversation_start_time[user_id]
    elapsed_time = current_time - start_time

    if elapsed_time > 600:
        generated_response = "Thanks for the exchange, I have to go soon! Bye."        
        audio_file_path, duration = await text_to_speech_audio(generated_response, voice)
        audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)  
        await asyncio.to_thread(delete_audio_file, audio_file_path)      
        conversation_history.pop(user_id, None)
        conversation_start_time.pop(user_id, None)
        
        return {
            "user_input": user_input,
            "generated_response": generated_response,
            "audio_base64": audio_base64,
            "conversation_history": user_history,
            "user_audio_base64": user_audio_base64,
            "metrics": {
                "transcription_time": transcription_time,
            }
        }

    if user_input == "thank you.":
        conversation_history.pop(user_id, None)
        conversation_start_time.pop(user_id, None)
        
        return {
            "user_input": user_input,
            "generated_response": None,
            "audio_base64": None,
            "conversation_history": None,
            "user_audio_base64": user_audio_base64,
            "metrics": {
                "transcription_time": transcription_time,
            }
        }

    if language in ['fr', 'unknown']:
        generated_response = "Hum.. I don't speak French, please say it in English."      
        audio_file_path, duration = await text_to_speech_audio(generated_response, voice)
        audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
        await asyncio.to_thread(delete_audio_file, audio_file_path)
        log_conversation_and_message(db, user_id, category, user_input, user_input, generated_response, user_audio_base64, audio_base64, marker="french_responses", ia_audio_duration=duration)
        
        return {
            "user_input": user_input,
            "generated_response": generated_response,
            "audio_base64": audio_base64,
            "conversation_history": None,
            "user_audio_base64": user_audio_base64,
            "metrics": {
                "transcription_time": transcription_time,
            }
        }

    is_quality_sentence = evaluate_sentence_quality_phi3(user_input)
    print("fin sentence")
    
    if not is_quality_sentence:
        generated_response = "I didn't understand that. Could you please rephrase?"
        suggestion_prompt = f"Improve the following unclear response in one sentence easy: {user_input}"
        suggestion = await asyncio.to_thread(generate_phi3_response, suggestion_prompt)       
        audio_file_path, duration = await text_to_speech_audio(generated_response, voice)
        audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
        await asyncio.to_thread(delete_audio_file, audio_file_path)
        log_conversation_and_message(db, user_id, category, user_input, user_input, generated_response, user_audio_base64, audio_base64, marker="unclear_responses", suggestion=suggestion, ia_audio_duration=duration)
        
        return {
            "user_input": user_input,
            "generated_response": generated_response,
            "audio_base64": audio_base64,
            "conversation_history": None,
            "user_audio_base64": user_audio_base64,
            "suggestion": suggestion,
            "metrics": {
                "transcription_time": transcription_time,
            }
        }

    
    generated_response = await asyncio.to_thread(generate_ai_response_alice, user_input, user_history)  
    print(f"Réponse générée en secondes")  
    user_history.append({'input': user_input, 'response': generated_response})
    audio_file_path, duration = await text_to_speech_audio(generated_response, voice)
    audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
    log_conversation_and_message(db, user_id, category, user_input, user_input, generated_response, user_audio_base64, audio_base64, ia_audio_duration=duration)
    await asyncio.to_thread(delete_audio_file, user_audio_path)
    await asyncio.to_thread(delete_audio_file, audio_file_path)
    
    
    return {
        "user_input": user_input,
        "generated_response": generated_response,
        "audio_base64": audio_base64,
        "conversation_history": user_history,
        "user_audio_base64": user_audio_base64,
        "metrics": {
            "transcription_time": transcription_time,
        }
    }


@app.post("/chat_repeat/start")
async def start_chat(request_data: StartChatRequest, voice: str = "english_ljspeech_tacotron2-DDC", db: Session = Depends(get_db), current_user: int = Depends(get_current_user)):
    global current_prompt
    user_id = current_user.id
    choice = request_data.choice    
    prompt = get_prompt(choice)
    current_prompt = prompt.rstrip("!?.")       
    generated_response = generate_response_variation(prompt)        
    audio_file_path = text_to_speech_audio(generated_response, voice) 
    audio_base64 = file_to_base64(audio_file_path)        
    log_conversation_to_db(db, user_id, current_prompt, generated_response)
    conversation = create_or_get_conversation(db, user_id, choice)
    log_message_to_db(db, user_id, conversation.id, current_prompt, generated_response, audio_base64)
        
    return {
            "generated_response": generated_response,
            "audio_base64": audio_base64
    }

@app.post("/chat_repeat")
async def chat_with_brain(choice: str = Form(...), audio_file: UploadFile = File(...), voice: str = "english_ljspeech_tacotron2-DDC", db: Session = Depends(get_db), current_user: int = Depends(get_current_user)):
    global current_prompt
    user_id = current_user.id
    category = get_category(choice)
    user_audio_path = f"./audio/user/{audio_file.filename}"
    with open(user_audio_path, "wb") as f:
        f.write(await audio_file.read())
        
    user_input = transcribe_audio(user_audio_path).strip().lower()
    user_input = user_input.rstrip("!?.")    
    user_audio_base64 = file_to_base64(user_audio_path)
    print(f"User's voice input: {user_input}")
    expected_prompt = current_prompt.lower().strip()
    generated_response, current_prompt = handle_response(user_input, expected_prompt, category)    
    audio_file_path = text_to_speech_audio(generated_response, voice)
    audio_base64 = file_to_base64(audio_file_path)    
    log_conversation_and_message(db, user_id, category, current_prompt, user_input, generated_response, user_audio_base64, audio_base64)

    return {
        "user_input": user_input,
        "generated_response": generated_response,
        "audio_base64": audio_base64,
        "user_audio_base64": user_audio_base64
    }

@app.post("/chat_repeat/stop")
async def stop_chat(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        user_id = current_user.id
        active_conversation = db.query(Conversation).filter(
            ((Conversation.user1_id == user_id) | (Conversation.user2_id == user_id)),
            Conversation.start_time.isnot(None),
            Conversation.end_time.is_(None)
        ).first()

        if active_conversation:
            active_conversation.end_session(db)
            return {"message": "Chat session ended successfully.", "conversation_id": active_conversation.id}
        else:
            return {"error": "No active chat session found or already ended."}

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/translate")
async def translate_message(request: TranslationRequest):
    try:
        prompt = f"traduire ce message en français : \n {request.message}"
        generated_response = generate_phi3_response(prompt)

        return {
            "translated_message": generated_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))   

@app.post("/end_conversation/{conversation_id}")
def end_conversation(conversation_id: int, db: Session = Depends(get_db)):
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conversation:
        conversation.end_session(db)
        return {"message": "Conversation ended successfully"}
    else:
        return {"error": "Conversation not found"}, 404

@app.get("/user/conversations", response_model=List[ConversationSchema])
def get_user_conversations(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conversations = db.query(Conversation).filter(
        (Conversation.user1_id == current_user.id) | (Conversation.user2_id == current_user.id)
    ).all()

    for conversation in conversations:
        conversation.start_time = conversation.start_time.replace(tzinfo=timezone.utc).isoformat()

    for conversation in conversations:
        conversation.category = category_mapping.get(conversation.category, conversation.category)

    return conversations

@app.post("/chat/message")
async def send_message(request: ChatRequest):
    try:
        prompt = request.message
        generated_response = generate_phi3_response(prompt)
        audio_file_path = text_to_speech_audio(generated_response, "english_ljspeech_tacotron2-DDC")
        audio_base64 = file_to_base64(audio_file_path)

        return {
            "generated_response": generated_response,
            "audio_base64": audio_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/r_a_m/{conversation_id}/messages", response_model=List[MessageSchema])
def get_conversation_messages(
    conversation_id: int,
    category: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        ((Conversation.user1_id == current_user.id) | (Conversation.user2_id == current_user.id))
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    query = db.query(Message).filter(Message.conversation_id == conversation.id).order_by(asc(Message.timestamp))

    if category:
        query = query.filter(Message.category == category)

    messages = query.all()
    print(messages)
    return messages


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_main:app", host="0.0.0.0", port=8000, reload=True)
