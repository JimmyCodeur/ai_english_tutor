from fastapi import FastAPI, Depends, Form, HTTPException, status, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import asc
from bdd.schema_pydantic import UserCreate, User, CreateAdminRequest as PydanticUser
from bdd.schema_pydantic import TokenInBody, UserLoginResponse, TranslationRequest, ChatRequest, ConversationSchema, MessageSchema, StartChatRequest, AnalysisResponse
from pydantic import EmailStr, constr, BaseModel
from datetime import date
from bdd.database import get_db
from bdd.crud import create_user, delete_user, update_user, get_user_by_id, update_user_role, log_conversation_to_db, log_message_to_db, log_conversation_and_message
from bdd.models import Message, User as DBUser, Role
from bdd.models import Conversation
from token_security import get_current_user, authenticate_user, create_access_token, revoked_tokens
from log_conversation import log_conversation, create_or_get_conversation
from load_model import generate_phi3_response, generate_ollama_response
from help_suggestion import help_sugg
from tts_utils import text_to_speech_audio
from transcribe_audio import transcribe_audio
from audio_utils import file_to_base64
from detect_langs import detect_language
from cors import add_middleware
from prompt import generate_response_variation, get_prompt, get_category, handle_response, category_mapping
from TTS.api import TTS
import nltk
from fastapi import FastAPI
from typing import List ,Optional
from datetime import timezone
import time
import re

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

@app.get("/analyze_session/{conversation_id}", response_model=AnalysisResponse)
def analyze_session(conversation_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    translations = db.query(Message).filter(
        Message.conversation_id == conversation_id,
        Message.user_input.like("how did you say in english%")
    ).all()
    print(translations)

    unclear_responses = db.query(Message).filter(
        Message.conversation_id == conversation_id,
        Message.response == "I didn't understand that. Could you please rephrase?"
    ).all()

    french_responses = db.query(Message).filter(
        Message.conversation_id == conversation_id,
        Message.response == "Hum.. I don't speak French, please say it in English."
    ).all()

    analysis_result = {
        "translations": [msg.content for msg in translations],
        "unclear_responses": [msg.content for msg in unclear_responses],
        "french_responses": [msg.content for msg in french_responses]
    }
    
    return analysis_result

conversation_history = {}
conversation_start_time = {}
alice_start_greetings = "Hi! My name is Alice. What's your name?"

@app.post("/chat_conv/start")
async def chat_with_brain(request_data: StartChatRequest, voice: str = "english_ljspeech_tacotron2-DDC", db: Session = Depends(get_db), current_user: int = Depends(get_current_user)):
    global conversation_history
    global conversation_start_time

    user_id = current_user.id
    choice = request_data.choice

    audio_file_path = text_to_speech_audio(alice_start_greetings, voice)
    audio_base64 = file_to_base64(audio_file_path)

    # Initialize or retrieve conversation history for the user
    if user_id in conversation_history:
        user_history = conversation_history[user_id]
    else:
        user_history = []
        conversation_history[user_id] = user_history
        conversation_start_time[user_id] = time.time()

    # Log the initial conversation
    log_conversation_to_db(db, user_id, alice_start_greetings, alice_start_greetings) 
    conversation = create_or_get_conversation(db, user_id, choice)
    log_message_to_db(db, user_id, conversation.id, alice_start_greetings, alice_start_greetings, audio_base64)

    return {
        "generated_response": alice_start_greetings,
        "audio_base64": audio_base64
    }

import spacy

# Load the spaCy model for NER
nlp = spacy.load("en_core_web_sm")

def replace_french_locations(sentence: str) -> str:
    doc = nlp(sentence)
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:  # GPE (Geopolitical Entity), LOC (Location)
            sentence = sentence.replace(ent.text, "[LOCATION]")
    return sentence

def evaluate_sentence_quality_phi3(sentence: str) -> bool:
    # Replace French locations with a placeholder
    cleaned_sentence = replace_french_locations(sentence)
    print(cleaned_sentence)
    
    prompt = f"Is the following sentence well-constructed and grammatically correct? Answer with 'yes' or 'no':\n\n{cleaned_sentence}"
    model_name = 'phi3'
    response = generate_ollama_response(model_name, prompt).strip().lower()
    return response.startswith("yes")

def detect_translation_request(user_input: str) -> Optional[str]:
    # Regular expression to detect translation requests like "how do you say in english"
    match = re.search(r'how (do|did|can|could|would|will|should) (you|i|we) say[,]? in english[,]?\s*(.*)', user_input, re.IGNORECASE)
    if match and match.group(3):
        return match.group(3).strip()  # Return the phrase to be translated
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

    user_audio_path = f"./audio/user/{audio_file.filename}"
    with open(user_audio_path, "wb") as f:
        f.write(await audio_file.read())

    user_input = transcribe_audio(user_audio_path).strip().lower()
    print(user_input)
    user_audio_base64 = file_to_base64(user_audio_path)

    phrase_to_translate = detect_translation_request(user_input)
    print(phrase_to_translate)
    if phrase_to_translate:
        prompt = f"Traduire cette phrase en anglais sans ajouter des commentaires: {phrase_to_translate}"
        translated_phrase = generate_phi3_response(prompt)
        generated_response = f"{translated_phrase}"
        audio_file_path = text_to_speech_audio(generated_response, voice)
        audio_base64 = file_to_base64(audio_file_path)
        log_conversation_and_message(db, user_id, category, user_input, user_input, generated_response, user_audio_base64, audio_base64, marker="translations")
        return {
            "user_input": user_input,
            "generated_response": generated_response,
            "audio_base64": audio_base64,
            "conversation_history": None,
            "user_audio_base64": user_audio_base64
        }
    
    if user_id in conversation_history:
        user_history = conversation_history[user_id]
    else:
        user_history = []
        conversation_history[user_id] = user_history
        conversation_start_time[user_id] = time.time() 

    if not user_history and user_input == alice_start_greetings:
        pass
    else:
        user_history.append(user_input)
    
    language = detect_language(user_input)  

    if time.time() - conversation_start_time[user_id] > 60:  # 900 seconds = 15 minutes
        print(conversation_start_time)
        generated_response = "Thanks for the exchange, I have to go soon! Bye."
        audio_file_path = text_to_speech_audio(generated_response, voice)
        audio_base64 = file_to_base64(audio_file_path)
        return {
            "user_input": user_input,
            "generated_response": generated_response,
            "audio_base64": audio_base64,
            "conversation_history": user_history,
            "user_audio_base64": user_audio_base64
        }

    # Handle thank you case
    if user_input == "thank you.":
        return {
            "user_input": user_input,
            "generated_response": None,
            "audio_base64": None,
            "conversation_history": None,
            "user_audio_base64": user_audio_base64
        }
    
            # Check if user input is in French
    if language in ['fr', 'unknown']:
        generated_response = "Hum.. I don't speak French, please say it in English."
        audio_file_path = text_to_speech_audio(generated_response, voice)
        audio_base64 = file_to_base64(audio_file_path)
        log_conversation_and_message(db, user_id, category, user_input, user_input, None, user_audio_base64, None, marker="french_responses")
        return {
            "user_input": user_input,
            "generated_response": generated_response,
            "audio_base64": audio_base64,
            "conversation_history": None,
            "user_audio_base64": user_audio_base64
        }
    

        # Check if user input is valid
    if not evaluate_sentence_quality_phi3(user_input):
        generated_response = "I didn't understand that. Could you please rephrase?"
        audio_file_path = text_to_speech_audio(generated_response, voice)
        audio_base64 = file_to_base64(audio_file_path)
        log_conversation_and_message(db, user_id, category, user_input, user_input, generated_response, user_audio_base64, audio_base64, marker="unclear_responses")
        return {
            "user_input": user_input,
            "generated_response": generated_response,
            "audio_base64": audio_base64,
            "conversation_history": None,
            "user_audio_base64": user_audio_base64
        }

    def generate_ai_response_alice(previous_input, user_history, context_sentences=2):
        # Alice's biography
        alice_intro = "You are alice" 
        alice_main = "You are alice. your goal is just to answer the phrase naturally user without asking questions. Remember that you speak with a beginner in English and that you must use very simple sentences"

        conversation_context = " ".join(user_history)
        print(user_history)

        if not user_history:
            # If it's the first interaction, use the full introduction
            full_prompt = f"{alice_intro}{previous_input}\n\n{conversation_context}"
        else:
            full_prompt = f"{alice_main}{previous_input}\n\n{conversation_context}"

        model_name = 'phi3'
        generated_response = generate_ollama_response(model_name, full_prompt)

        if isinstance(generated_response, list):
            generated_response = ' '.join(generated_response)

        sentences = nltk.tokenize.sent_tokenize(generated_response)
        limited_response = ' '.join(sentences[:context_sentences])  

        return limited_response
    
    conversation_topics = [
        ("{last_response} Nice to meet you. How are you?", generate_ai_response_alice),
        ("{last_response} Where do you come from?", generate_ai_response_alice),
        ("{last_response} I’m from an English-speaking country. How about you?", generate_ai_response_alice),
        ("{last_response} That's interesting. What brings you here?", generate_ai_response_alice),
        ("{last_response} That's great! How long have you been studying English?", generate_ai_response_alice),
        ("{last_response} Do you live in this city?", generate_ai_response_alice),
        ("{last_response} I just arrived here a few days ago. How long have you been here?", generate_ai_response_alice),
        ("{last_response} What do you do during the day?", generate_ai_response_alice),
        ("{last_response} I work as a teacher. And you?", generate_ai_response_alice),
        ("{last_response} What do you enjoy doing in your free time?", generate_ai_response_alice),
        ("{last_response} Do you have any hobbies?", generate_ai_response_alice),
        ("{last_response} Tell me about your family.", generate_ai_response_alice),
        ("{last_response} Do you have brothers or sisters?", generate_ai_response_alice),
        ("{last_response} What’s your favorite dish?", generate_ai_response_alice),
        ("{last_response} I love trying different cuisines. What about you?", generate_ai_response_alice),
        ("{last_response} Do you enjoy traveling?", generate_ai_response_alice),
        ("{last_response} What’s your favorite place you’ve visited?", generate_ai_response_alice),
        ("{last_response} I would love to visit England. Any recommendations?", generate_ai_response_alice),
        ("{last_response} Do you speak any other languages?", generate_ai_response_alice),
        ("{last_response} I’m learning English. Do you want to practice together?", generate_ai_response_alice),
        ("{last_response} It was nice talking to you!", generate_ai_response_alice),
        ("{last_response} I hope we can chat again soon.", generate_ai_response_alice)
    ]

    user_response_index = len(user_history) - 1
    ai_prompt, response_function = conversation_topics[user_response_index]

    if response_function:
        last_user_response = user_history[-1] if user_history else ""
        initial_response = generate_ai_response_alice(last_user_response, user_history)
        ai_prompt = ai_prompt.format(last_response=initial_response)

    generated_response = ai_prompt

    

    audio_file_path = text_to_speech_audio(generated_response, voice)
    audio_base64 = file_to_base64(audio_file_path)



    log_conversation_and_message(db, user_id, category, ai_prompt, user_input, generated_response, user_audio_base64, audio_base64)
    return {
        "user_input": user_input,
        "generated_response": generated_response,
        "audio_base64": audio_base64,
        "conversation_history": user_history,
        "user_audio_base64": user_audio_base64
    }

@app.post("/chat_repeat/start")
async def start_chat(request_data: StartChatRequest, voice: str = "english_ljspeech_tacotron2-DDC", db: Session = Depends(get_db), current_user: int = Depends(get_current_user)):
    global current_prompt
    user_id = current_user.id
    choice = request_data.choice    
    prompt = get_prompt(choice)
    current_prompt = prompt.rstrip("!?.")
    print(prompt)        
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
        log_conversation(prompt, generated_response)
        return {
            "generated_response": generated_response,
            "audio_base64": audio_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/r_a_m/{conversation_id}/messages", response_model=List[MessageSchema])
def get_conversation_messages(
    conversation_id: int,
    category: Optional[str] = None,  # Ajouter une query parameter pour la catégorie
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        ((Conversation.user1_id == current_user.id) | (Conversation.user2_id == current_user.id))
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Créer une requête de base pour filtrer par conversation_id et trier par timestamp ascendant
    query = db.query(Message).filter(Message.conversation_id == conversation.id).order_by(asc(Message.timestamp))

    # Filtrer par catégorie si une catégorie est spécifiée
    if category:
        query = query.filter(Message.category == category)

    # Exécuter la requête et récupérer les résultats
    messages = query.all()
    print(messages)
    return messages


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_main:app", host="127.0.0.1", port=8000, reload=True)