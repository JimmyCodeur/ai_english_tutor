#fichier api_main.py
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
from back.prompt.conversation_personalities import get_ai_function_for_choice, get_category, get_character_greeting
from pydantic import EmailStr, constr, BaseModel
from back.token_security import get_current_user, authenticate_user, create_access_token, revoked_tokens
from back.load_model import generate_phi3_response, generate_ollama_response
from back.help_suggestion import help_sugg
from back.tts_utils import text_to_speech_audio, ensure_audio_directories, get_best_voice_for_character
from back.edge_tts import edge_text_to_speech_with_fallback
from back.transcribe_audio import transcribe_audio
from back.audio_utils import file_to_base64, is_valid_audio_file, delete_audio_file
from back.detect_langs import detect_language
from back.cors import add_middleware
from back.bdd.database import init_db, check_db_health
from typing import List, Optional
from datetime import timezone, timedelta, date, datetime
import nltk
import time
import re
import spacy
import aiofiles
import asyncio
import os
import random

app = FastAPI()
add_middleware(app)
app.mount("/static", StaticFiles(directory="./front/static"), name="static")
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

conversation_history = {}
conversation_start_time = {}

# 🆕 MAPPING DES CHOIX VERS LES ENDPOINTS SPÉCIFIQUES - AVEC NOUVEAUX PERSONNAGES
CHOICE_TO_CHARACTER = {
    'conv_greetings_common_conversations': 'alice',
    'conv_taxi_booking': 'mike',
    'conv_airport_ticket': 'sarah',
    # 🆕 NOUVEAUX PERSONNAGES DÉBUTANTS
    'conv_beginner_teacher': 'emma',
    'conv_simple_restaurant': 'tom',
    'conv_simple_shopping': 'lucy',
    'conv_simple_directions': 'ben'
}

# 🆕 MAPPING POUR LES INFORMATIONS DES PERSONNAGES - AVEC NOUVEAUX
CHARACTER_INFO = {
    'alice': {
        'name': 'Alice',
        'role': 'Conversation Partner',
        'description': 'Friendly English conversation partner for general practice'
    },
    'mike': {
        'name': 'Mike',
        'role': 'NYC Taxi Dispatcher',
        'description': 'NYC taxi dispatcher for booking practice'
    },
    'sarah': {
        'name': 'Sarah',
        'role': 'Airline Customer Service',
        'description': 'Airline customer service for flight booking practice'
    },
    # 🆕 NOUVEAUX PERSONNAGES POUR DÉBUTANTS
    'emma': {
        'name': 'Emma',
        'role': 'English Teacher for Beginners',
        'description': 'Patient English teacher specializing in absolute beginners'
    },
    'tom': {
        'name': 'Tom',
        'role': 'Friendly Waiter',
        'description': 'Simple restaurant waiter for basic food ordering practice'
    },
    'lucy': {
        'name': 'Lucy',
        'role': 'Shop Assistant',
        'description': 'Helpful shop assistant for basic shopping conversations'
    },
    'ben': {
        'name': 'Ben',
        'role': 'Tourist Guide',
        'description': 'Patient tourist guide for asking directions and basic city info'
    }
}

# 🆕 MAPPING POUR LES CATÉGORIES - AVEC NOUVELLES
category_mapping = {
    'conv_greetings_common_conversations': 'Conversations générales',
    'conv_taxi_booking': 'Réservation de taxi',
    'conv_airport_ticket': 'Réservation de vol',
    # 🆕 NOUVELLES CATÉGORIES DÉBUTANTS
    'conv_beginner_teacher': 'Professeure pour débutants',
    'conv_simple_restaurant': 'Restaurant simple',
    'conv_simple_shopping': 'Shopping basique',
    'conv_simple_directions': 'Demander son chemin'
}

# Mapping pour retenir les choix actifs par utilisateur
active_user_choices = {}

# 🆕 CONFIGURATION EDGE-TTS PAR DÉFAUT - AVEC NOUVEAUX PERSONNAGES
DEFAULT_EDGE_VOICES = {
    "alice": "edge_alice",
    "mike": "edge_mike",
    "sarah": "edge_sarah",
    # 🆕 NOUVEAUX PERSONNAGES
    "emma": "edge_emma",
    "tom": "edge_tom",
    "lucy": "edge_lucy",
    "ben": "edge_ben"
}

# Événement de démarrage pour initialiser les répertoires
@app.on_event("startup")
async def startup_event():
    print("🚀 Initialisation de TalkAI...")
    
    # Vérifier les variables d'environnement
    required_env = ['POSTGRES_HOST', 'POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_DB']
    missing_env = [var for var in required_env if not os.getenv(var)]
    if missing_env:
        print(f"❌ Variables d'environnement manquantes: {missing_env}")
        sys.exit(1)
    
    # Initialiser la base de données
    try:
        init_db()
        if check_db_health():
            print("✅ Base de données opérationnelle")
        else:
            print("⚠️ Problème de santé de la DB")
    except Exception as e:
        print(f"❌ Erreur initialisation DB: {e}")
        # Ne pas arrêter l'app, continuer avec les autres services
    
    # Créer les dossiers nécessaires
    directories = [
        "./front/static/assets/images/avatars",
        "./audio/user", 
        "./audio/ia",
        "./logs"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("📁 Dossiers créés")
    
    # Initialiser les répertoires audio
    ensure_audio_directories()
    print("✅ Répertoires audio initialisés")
    
    # Test Edge-TTS
    try:
        print("🧪 Test Edge-TTS...")
        test_audio_path, _ = await edge_text_to_speech_with_fallback("Hello", "alice")
        print(f"✅ Edge-TTS opérationnel")
        if os.path.exists(test_audio_path):
            os.remove(test_audio_path)
    except Exception as e:
        print(f"⚠️ Edge-TTS indisponible: {e}")
    
    print("🎉 TalkAI initialisé avec succès!")

# ===== ROUTES PAGES WEB =====
@app.get('/')
def index():
    return FileResponse('./front/form-login.html')

@app.get("/test.html")
def get_test():
    return FileResponse('./front/test.html')

@app.get("/users-profile.html")
def get_users_profile():
    return FileResponse('./front/users-profile.html')

@app.get("/form-register.html")
def get_form_register():
    return FileResponse('./front/form-register.html')

@app.get("/form-login.html")
def get_form_login():
    return FileResponse('./front/form-login.html')

@app.get("/profile-setting.html")
def get_profile_setting():
    return FileResponse('./front/profile-setting.html')

@app.get("/brain-info-course.html")
def get_brain_info_course():
    return FileResponse('./front/brain-info-course.html')

@app.get("/home.html")
def get_home():
    return FileResponse('./front/home.html')

@app.get("/conversation.html")
def get_conversation():
    return FileResponse('./front/conversation.html')

@app.get("/analysis.html")
def get_analysis():
    return FileResponse('./front/analysis.html')

@app.get("/voice-tester.html")
def get_voice_tester():
    return FileResponse('./front/voice-tester.html')

@app.get("/course.html")
def get_course():
    return FileResponse('./front/course.html')

# ===== ROUTES UTILISATEURS =====
@app.get("/users/me", response_model=User)
def read_current_user(current_user: DBUser = Depends(get_current_user)):
    return User(
        id=current_user.id,
        email=current_user.email,
        nom=current_user.nom,
        date_naissance=current_user.date_naissance,
        date_creation=current_user.date_creation if hasattr(current_user, 'date_creation') else datetime.now(),
        role=current_user.role,
        consent=current_user.consent or False
    )

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
    consent: Optional[bool] = Form(False),
    db: Session = Depends(get_db)
):
    print(f"Reçu : email={email}, nom={nom}, date_naissance={date_naissance}, password={password}, consent={consent}")
    user_data = UserCreate(
        email=email,
        nom=nom,
        date_naissance=date_naissance,
        password=password,
        consent=consent
    )
    try:
        db_user = create_user(db=db, user=user_data)
        return User(
            id=db_user.id,
            email=db_user.email,
            nom=db_user.nom,
            date_naissance=db_user.date_naissance,
            date_creation=db_user.date_creation,
            role=db_user.role,
            consent=db_user.consent or False
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

@app.post("/token", response_model=dict)
def login_for_access_token_alt(
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
    if current_user.role != Role.ADMIN.value:
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

# ===== FONCTIONS UTILITAIRES =====
def calculate_avg_user_response_time(messages):
    """Calcule le temps de réponse moyen de l'utilisateur avec gestion d'erreur"""
    try:
        if not messages or len(messages) < 2:
            return "N/A"
        
        # Trier les messages par timestamp
        sorted_messages = sorted(messages, key=lambda msg: msg.timestamp)
        
        time_differences = []
        for i in range(1, len(sorted_messages)):
            try:
                current_msg = sorted_messages[i]
                previous_msg = sorted_messages[i-1]
                
                # Vérifier que les messages ont des timestamps valides
                if not current_msg.timestamp or not previous_msg.timestamp:
                    continue
                
                # Calculer la différence de temps
                time_diff = (current_msg.timestamp - previous_msg.timestamp).total_seconds()
                
                # Soustraire la durée audio de l'IA si disponible
                if hasattr(previous_msg, 'ia_audio_duration') and previous_msg.ia_audio_duration:
                    time_diff -= previous_msg.ia_audio_duration
                
                # Ignorer les différences négatives ou trop grandes (> 5 minutes)
                if 0 < time_diff < 300:
                    time_differences.append(time_diff)
                    
            except Exception as e:
                print(f"⚠️ Erreur calcul temps réponse message {i}: {e}")
                continue
        
        if not time_differences:
            return "N/A"
        
        avg_response_time = sum(time_differences) / len(time_differences)
        avg_response_time_delta = timedelta(seconds=avg_response_time)
        
        # Formater le résultat
        total_seconds = int(avg_response_time_delta.total_seconds())
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
            
    except Exception as e:
        print(f"❌ Erreur globale calcul temps réponse: {e}")
        return "N/A"

def evaluate_sentence_quality_phi3(sentence: str) -> bool:
    print("🔍 Début évaluation phrase")
    prompt = f"Is the following sentence understandable and acceptable for simple conversation, ignoring small grammar mistakes, capitalization, and punctuation? Answer 'yes' or 'no' only:\n\n{sentence}"
    print(prompt)
    model_name = 'phi3'
    response = generate_ollama_response(model_name, prompt).strip().lower()
    response_short = response.split(".")[0].strip()
    print(f"📊 Response from model: {response_short}")
    if "yes" in response_short or "understandable" in response_short:
        return True
    elif "no" in response_short:
        return False
    else:
        log_error(f"Unexpected response format from model: {response}")
        print("⚠️ Unexpected response format from model.")
        return False

def detect_translation_request(user_input: str) -> Optional[str]:
    match = re.search(r'how (do|did|can|could|would|will|should) (you|i|we) say[,]? in english[,]?\s*(.*)', user_input, re.IGNORECASE)
    if match and match.group(3):
        return match.group(3).strip()
    return None

async def process_audio_and_transcribe(audio_file: UploadFile):
    """Fonction helper pour traiter l'audio et la transcription"""
    if not is_valid_audio_file(audio_file):
        raise HTTPException(status_code=400, detail="Invalid audio file format")
    
    user_audio_path = f"./audio/user/{audio_file.filename}"
    async with aiofiles.open(user_audio_path, "wb") as f:
        await f.write(await audio_file.read())
    
    print("🎤 Début transcription audio")
    transcription_task = asyncio.create_task(transcribe_audio(user_audio_path))
    user_audio_base64 = await asyncio.to_thread(file_to_base64, user_audio_path)    
    user_input, transcription_time = await transcription_task
    print(f"✅ Transcription terminée: '{user_input}'")
    
    await asyncio.to_thread(delete_audio_file, user_audio_path)
    return user_input.strip().lower(), user_audio_base64, transcription_time

async def handle_common_responses(user_input: str, character: str, db: Session, user_id: int, category: str, user_audio_base64: str):
    """Gère les réponses communes avec Phi3 naturel"""
    
    # Gestion des demandes de traduction
    phrase_to_translate = detect_translation_request(user_input)
    if phrase_to_translate:
        prompt = f"""Translate this phrase into natural English: "{phrase_to_translate}"
        
        Give only the English translation, no explanations."""
        
        start_time = time.time()
        translated_phrase = await asyncio.to_thread(generate_phi3_response, prompt)
        end_time = time.time()
        log_response_time_phi3(start_time, end_time)
        
        # Réponse naturelle pour la traduction
        if character == 'emma':
            generated_response = f"In English, we say: {translated_phrase}. Can you try saying it?"
        elif character == 'mike':
            generated_response = f"In English, that would be: {translated_phrase}."
        elif character == 'sarah':
            generated_response = f"The English phrase is: {translated_phrase}."
        else:
            generated_response = f"In English, you can say: {translated_phrase}."
        
        audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, character)
        audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
        await asyncio.to_thread(delete_audio_file, audio_file_path)
        
        log_conversation_and_message(db, user_id, category, user_input, user_input, generated_response, user_audio_base64, audio_base64, marker="translations", ia_audio_duration=duration)
        
        return {
            "user_input": user_input,
            "generated_response": generated_response,
            "audio_base64": audio_base64,
            "user_audio_base64": user_audio_base64,
            "type": "translation"
        }
    
    # Détection de langue française
    language = detect_language(user_input)
    if language in ['fr', 'unknown']:
        # Réponses naturelles selon le personnage
        if character == 'emma':
            generated_response = "I don't speak French. Can you try in English? Even simple words are good!"
        elif character == 'mike':
            generated_response = "Sorry, I only speak English here at the dispatch. Can you try in English?"
        elif character == 'sarah':
            generated_response = "I need to speak in English to help you with your booking. Can you try in English?"
        else:
            generated_response = "I don't speak French, could you please try in English?"
        
        audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, character)
        audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
        await asyncio.to_thread(delete_audio_file, audio_file_path)
        
        log_conversation_and_message(db, user_id, category, user_input, user_input, generated_response, user_audio_base64, audio_base64, marker="french_responses", ia_audio_duration=duration)
        
        return {
            "user_input": user_input,
            "generated_response": generated_response,
            "audio_base64": audio_base64,
            "user_audio_base64": user_audio_base64,
            "type": "french_response"
        }
    
    # Évaluation de la qualité avec réponse naturelle
    is_quality_sentence = evaluate_sentence_quality_phi3(user_input)
    if not is_quality_sentence:
        # Réponses naturelles d'incompréhension
        if character == 'emma':
            generated_response = "I didn't understand. Can you say it again with simple words?"
        elif character == 'mike':
            generated_response = "Sorry, I didn't catch that. Can you repeat it for me?"
        elif character == 'sarah':
            generated_response = "I'm sorry, I didn't understand. Could you please rephrase that?"
        else:
            generated_response = "I'm sorry, I didn't quite understand. Could you say that again?"
        
        # Générer une suggestion d'amélioration
        suggestion_prompt = f"""The user said: "{user_input}"
        
        This was unclear. Suggest a clearer way to say the same thing in simple English.
        Give only the improved sentence, no explanations."""
        
        suggestion = await asyncio.to_thread(generate_phi3_response, suggestion_prompt)
        
        audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, character)
        audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
        await asyncio.to_thread(delete_audio_file, audio_file_path)
        
        log_conversation_and_message(db, user_id, category, user_input, user_input, generated_response, user_audio_base64, audio_base64, marker="unclear_responses", suggestion=suggestion, ia_audio_duration=duration)
        
        return {
            "user_input": user_input,
            "generated_response": generated_response,
            "audio_base64": audio_base64,
            "user_audio_base64": user_audio_base64,
            "suggestion": suggestion,
            "type": "unclear_response"
        }
    
    return None  # Aucune réponse commune trouvée

# ===== ENDPOINTS PERSONNAGES EXISTANTS =====

@app.post("/chat_conv/alice")
async def chat_with_alice(
    audio_file: UploadFile = File(None),
    text_message: Optional[str] = Form(None),
    voice: str = "edge_alice",
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_user)
):
    """Conversation avec Alice - Support audio ET texte (EDGE-TTS)"""
    try:
        user_id = current_user.id
        choice = "conv_greetings_common_conversations"
        alice_key = f"{user_id}_alice"
        category = get_category(choice)
        character = "alice"
        print(f"🌟 Conversation avec Alice pour l'utilisateur {user_id} (Edge-TTS)")
        
        # Traitement AUDIO OU TEXTE
        if text_message and text_message.strip():
            print(f"📝 Message texte reçu: '{text_message}'")
            user_input = text_message.strip()
            user_audio_base64 = None
            transcription_time = 0
        elif audio_file and audio_file.size > 100:
            print(f"🎤 Message audio reçu (taille: {audio_file.size} bytes)")
            try:
                user_input, user_audio_base64, transcription_time = await process_audio_and_transcribe(audio_file)
            except Exception as transcribe_error:
                print(f"❌ Erreur transcription: {transcribe_error}")
                raise HTTPException(status_code=500, detail=f"Erreur de transcription: {str(transcribe_error)}")
        else:
            print(f"❌ Aucun message valide reçu")
            raise HTTPException(status_code=400, detail="Aucun message audio ou texte valide reçu")

        print(f"💬 Message traité: '{user_input}'")

        # Vérifier si c'est le TOUT PREMIER message
        is_very_first_message = alice_key not in conversation_history or len(conversation_history[alice_key]) == 0

        # Initialiser l'historique si nécessaire
        if alice_key not in conversation_history:
            conversation_history[alice_key] = []
            conversation_start_time[alice_key] = time.time()
            active_user_choices[user_id] = choice

        # Vérifier les réponses communes (sauf pour le premier message)
        if not is_very_first_message:
            try:
                common_response = await handle_common_responses(user_input, character, db, user_id, category, user_audio_base64)
                if common_response:
                    common_response["character"] = "Alice"
                    print(f"🔄 Réponse commune utilisée: {common_response['type']}")
                    return common_response
            except Exception as common_error:
                print(f"❌ Erreur réponses communes: {common_error}")

        # Vérifier timeout (10 minutes)
        current_time = time.time()
        start_time = conversation_start_time.get(alice_key, current_time)
        elapsed_time = current_time - start_time
        if elapsed_time > 600:
            print(f"⏰ Session expirée ({elapsed_time:.0f}s)")
            generated_response = "It was lovely chatting with you! I have to help other students now. Keep practicing your English!"        
            
            audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, character)
            audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
            await asyncio.to_thread(delete_audio_file, audio_file_path)
            
            conversation_history.pop(alice_key, None)
            conversation_start_time.pop(alice_key, None)
            active_user_choices.pop(user_id, None)
            
            return {
                "character": "Alice",
                "user_input": user_input,
                "generated_response": generated_response,
                "audio_base64": audio_base64,
                "user_audio_base64": user_audio_base64,
                "type": "session_end"
            }

        # Si c'est le tout premier message, gérer salutation + réponse
        if is_very_first_message:
            print(f"🎭 Premier message détecté - Génération de la salutation")
            try:
                # Générer la salutation
                greeting = get_character_greeting('alice', user_id)
                print(f"🎭 Message de salutation d'Alice: {greeting}")
                
                # Créer la conversation et sauvegarder la salutation
                conversation = create_or_get_conversation(db, user_id, choice)
                
                greeting_audio_path, greeting_duration = await edge_text_to_speech_with_fallback(greeting, character)
                greeting_audio_base64 = await asyncio.to_thread(file_to_base64, greeting_audio_path)
                await asyncio.to_thread(delete_audio_file, greeting_audio_path)
                
                log_message_to_db(
                    db, user_id, conversation.id, greeting,
                    greeting, greeting_audio_base64, ia_audio_duration=greeting_duration
                )
                
                # Ajouter la salutation à l'historique
                conversation_history[alice_key].append({'input': "", 'response': greeting})
                
                # Générer la réponse
                alice_history = conversation_history[alice_key]
                ai_function = get_ai_function_for_choice(choice)
                print(f"🤖 Génération de la réponse Alice pour: '{user_input}'")
                generated_response = await asyncio.to_thread(ai_function, user_input, alice_history, user_id)
                print(f"✅ Réponse générée par Alice: '{generated_response}'")
                
                # Ajouter à l'historique
                alice_history.append({'input': user_input, 'response': generated_response})
                conversation_history[alice_key] = alice_history
                
                # Générer audio pour la réponse
                audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, character)
                audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
                log_conversation_and_message(db, user_id, category, user_input, user_input, generated_response, user_audio_base64, audio_base64, ia_audio_duration=duration)
                await asyncio.to_thread(delete_audio_file, audio_file_path)
                
                return {
                    "character": "Alice",
                    "user_input": user_input,
                    "generated_response": generated_response,
                    "audio_base64": audio_base64,
                    "conversation_history": alice_history,
                    "user_audio_base64": user_audio_base64,
                    "greeting": greeting,
                    "greeting_audio": greeting_audio_base64,
                    "is_first_message": True,
                    "metrics": {
                        "transcription_time": transcription_time,
                    }
                }
            except Exception as greeting_error:
                print(f"❌ Erreur génération salutation: {greeting_error}")
                raise HTTPException(status_code=500, detail=f"Erreur de salutation: {str(greeting_error)}")

        # Messages suivants (conversation normale)
        print(f"💬 Message de conversation normale")
        try:
            ai_function = get_ai_function_for_choice(choice)
            alice_history = conversation_history.get(alice_key, [])
            print(f"🤖 Génération réponse Alice pour: '{user_input}'")
            generated_response = await asyncio.to_thread(ai_function, user_input, alice_history, user_id)
            print(f"✅ Réponse d'Alice: '{generated_response}'")
            
            # Ajouter à l'historique
            alice_history.append({'input': user_input, 'response': generated_response})
            conversation_history[alice_key] = alice_history
            
            # Générer audio
            audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, character)
            audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
            log_conversation_and_message(db, user_id, category, user_input, user_input, generated_response, user_audio_base64, audio_base64, ia_audio_duration=duration)
            await asyncio.to_thread(delete_audio_file, audio_file_path)
            
            return {
                "character": "Alice",
                "user_input": user_input,
                "generated_response": generated_response,
                "audio_base64": audio_base64,
                "conversation_history": alice_history,
                "user_audio_base64": user_audio_base64,
                "is_first_message": False,
                "metrics": {
                    "transcription_time": transcription_time,
                }
            }
        except Exception as response_error:
            print(f"❌ Erreur génération réponse: {response_error}")
            raise HTTPException(status_code=500, detail=f"Erreur de génération de réponse: {str(response_error)}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur générale avec Alice: {e}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An error occurred during the conversation with Alice: {str(e)}")

@app.post("/chat_conv/mike")
async def chat_with_mike(
    audio_file: UploadFile = File(None),
    text_message: Optional[str] = Form(None),
    voice: str = "edge_mike",
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_user)
):
    """Conversation avec Mike - Support audio ET texte (EDGE-TTS)"""
    try:
        user_id = current_user.id
        choice = "conv_taxi_booking"
        mike_key = f"{user_id}_mike"
        category = get_category(choice)
        character = "mike"
        print(f"🚕 Conversation avec Mike pour l'utilisateur {user_id} (Edge-TTS)")
        
        # Traitement AUDIO OU TEXTE
        if text_message and text_message.strip():
            print(f"📝 Message texte reçu: '{text_message}'")
            user_input = text_message.strip()
            user_audio_base64 = None
            transcription_time = 0
        elif audio_file and audio_file.size > 100:
            print(f"🎤 Message audio reçu (taille: {audio_file.size} bytes)")
            try:
                user_input, user_audio_base64, transcription_time = await process_audio_and_transcribe(audio_file)
            except Exception as transcribe_error:
                print(f"❌ Erreur transcription: {transcribe_error}")
                raise HTTPException(status_code=500, detail=f"Erreur de transcription: {str(transcribe_error)}")
        else:
            print(f"❌ Aucun message valide reçu")
            raise HTTPException(status_code=400, detail="Aucun message audio ou texte valide reçu")

        print(f"💬 Message traité: '{user_input}'")

        # Vérifier si c'est le TOUT PREMIER message
        is_very_first_message = mike_key not in conversation_history or len(conversation_history[mike_key]) == 0

        # Initialiser l'historique si nécessaire
        if mike_key not in conversation_history:
            conversation_history[mike_key] = []
            conversation_start_time[mike_key] = time.time()
            active_user_choices[user_id] = choice

        # Vérifier les réponses communes (sauf pour le premier message)
        if not is_very_first_message:
            try:
                common_response = await handle_common_responses(user_input, character, db, user_id, category, user_audio_base64)
                if common_response:
                    common_response["character"] = "Mike"
                    print(f"🔄 Réponse commune utilisée: {common_response['type']}")
                    return common_response
            except Exception as common_error:
                print(f"❌ Erreur réponses communes: {common_error}")

        # Vérifier timeout (10 minutes)
        current_time = time.time()
        start_time = conversation_start_time.get(mike_key, current_time)
        elapsed_time = current_time - start_time
        if elapsed_time > 600:
            print(f"⏰ Session expirée ({elapsed_time:.0f}s)")
            generated_response = "Thanks for calling NYC Taxi! I gotta take other calls now. Have a great ride!"        
            
            audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, character)
            audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
            await asyncio.to_thread(delete_audio_file, audio_file_path)
            
            conversation_history.pop(mike_key, None)
            conversation_start_time.pop(mike_key, None)
            active_user_choices.pop(user_id, None)
            
            return {
                "character": "Mike",
                "user_input": user_input,
                "generated_response": generated_response,
                "audio_base64": audio_base64,
                "user_audio_base64": user_audio_base64,
                "type": "session_end"
            }

        # Si c'est le tout premier message, gérer salutation + réponse
        if is_very_first_message:
            print(f"🎭 Premier message détecté - Génération de la salutation")
            try:
                # Générer la salutation
                greeting = get_character_greeting('mike', user_id)
                print(f"🎭 Message de salutation de Mike: {greeting}")
                
                # Créer la conversation et sauvegarder la salutation
                conversation = create_or_get_conversation(db, user_id, choice)
                
                greeting_audio_path, greeting_duration = await edge_text_to_speech_with_fallback(greeting, character)
                greeting_audio_base64 = await asyncio.to_thread(file_to_base64, greeting_audio_path)
                await asyncio.to_thread(delete_audio_file, greeting_audio_path)
                
                log_message_to_db(
                    db, user_id, conversation.id, greeting,
                    greeting, greeting_audio_base64, ia_audio_duration=greeting_duration
                )
                
                # Ajouter la salutation à l'historique
                conversation_history[mike_key].append({'input': "", 'response': greeting})
                
                # Générer la réponse
                mike_history = conversation_history[mike_key]
                ai_function = get_ai_function_for_choice(choice)
                print(f"🤖 Génération de la réponse Mike pour: '{user_input}'")
                generated_response = await asyncio.to_thread(ai_function, user_input, mike_history, user_id)
                print(f"✅ Réponse générée par Mike: '{generated_response}'")
                
                # Ajouter à l'historique
                mike_history.append({'input': user_input, 'response': generated_response})
                conversation_history[mike_key] = mike_history
                
                # Générer audio pour la réponse
                audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, character)
                audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
                log_conversation_and_message(db, user_id, category, user_input, user_input, generated_response, user_audio_base64, audio_base64, ia_audio_duration=duration)
                await asyncio.to_thread(delete_audio_file, audio_file_path)
                
                return {
                    "character": "Mike",
                    "user_input": user_input,
                    "generated_response": generated_response,
                    "audio_base64": audio_base64,
                    "conversation_history": mike_history,
                    "user_audio_base64": user_audio_base64,
                    "greeting": greeting,
                    "greeting_audio": greeting_audio_base64,
                    "is_first_message": True,
                    "metrics": {
                        "transcription_time": transcription_time,
                    }
                }
            except Exception as greeting_error:
                print(f"❌ Erreur génération salutation: {greeting_error}")
                raise HTTPException(status_code=500, detail=f"Erreur de salutation: {str(greeting_error)}")

        # Messages suivants (conversation normale)
        print(f"💬 Message de conversation normale")
        try:
            ai_function = get_ai_function_for_choice(choice)
            mike_history = conversation_history.get(mike_key, [])
            print(f"🤖 Génération réponse Mike pour: '{user_input}'")
            generated_response = await asyncio.to_thread(ai_function, user_input, mike_history, user_id)
            print(f"✅ Réponse de Mike: '{generated_response}'")
            
            # Ajouter à l'historique
            mike_history.append({'input': user_input, 'response': generated_response})
            conversation_history[mike_key] = mike_history
            
            # Générer audio
            audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, character)
            audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
            log_conversation_and_message(db, user_id, category, user_input, user_input, generated_response, user_audio_base64, audio_base64, ia_audio_duration=duration)
            await asyncio.to_thread(delete_audio_file, audio_file_path)
            
            return {
                "character": "Mike",
                "user_input": user_input,
                "generated_response": generated_response,
                "audio_base64": audio_base64,
                "conversation_history": mike_history,
                "user_audio_base64": user_audio_base64,
                "is_first_message": False,
                "metrics": {
                    "transcription_time": transcription_time,
                }
            }
        except Exception as response_error:
            print(f"❌ Erreur génération réponse: {response_error}")
            raise HTTPException(status_code=500, detail=f"Erreur de génération de réponse: {str(response_error)}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur générale avec Mike: {e}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An error occurred during the conversation with Mike: {str(e)}")

@app.post("/chat_conv/sarah")
async def chat_with_sarah(
    audio_file: UploadFile = File(None),
    text_message: Optional[str] = Form(None),
    voice: str = "edge_sarah",
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_user)
):
    """Conversation avec Sarah - Support audio ET texte (EDGE-TTS)"""
    try:
        user_id = current_user.id
        choice = "conv_airport_ticket"
        sarah_key = f"{user_id}_sarah"
        category = get_category(choice)
        character = "sarah"
        print(f"✈️ Conversation avec Sarah pour l'utilisateur {user_id} (Edge-TTS)")
        
        # Traitement AUDIO OU TEXTE
        if text_message and text_message.strip():
            print(f"📝 Message texte reçu: '{text_message}'")
            user_input = text_message.strip()
            user_audio_base64 = None
            transcription_time = 0
        elif audio_file and audio_file.size > 100:
            print(f"🎤 Message audio reçu (taille: {audio_file.size} bytes)")
            try:
                user_input, user_audio_base64, transcription_time = await process_audio_and_transcribe(audio_file)
            except Exception as transcribe_error:
                print(f"❌ Erreur transcription: {transcribe_error}")
                raise HTTPException(status_code=500, detail=f"Erreur de transcription: {str(transcribe_error)}")
        else:
            print(f"❌ Aucun message valide reçu")
            raise HTTPException(status_code=400, detail="Aucun message audio ou texte valide reçu")

        print(f"💬 Message traité: '{user_input}'")

        # Vérifier si c'est le TOUT PREMIER message
        is_very_first_message = sarah_key not in conversation_history or len(conversation_history[sarah_key]) == 0

        # Initialiser l'historique si nécessaire
        if sarah_key not in conversation_history:
            conversation_history[sarah_key] = []
            conversation_start_time[sarah_key] = time.time()
            active_user_choices[user_id] = choice

        # Vérifier les réponses communes (sauf pour le premier message)
        if not is_very_first_message:
            try:
                common_response = await handle_common_responses(user_input, character, db, user_id, category, user_audio_base64)
                if common_response:
                    common_response["character"] = "Sarah"
                    print(f"🔄 Réponse commune utilisée: {common_response['type']}")
                    return common_response
            except Exception as common_error:
                print(f"❌ Erreur réponses communes: {common_error}")

        # Vérifier timeout (10 minutes)
        current_time = time.time()
        start_time = conversation_start_time.get(sarah_key, current_time)
        elapsed_time = current_time - start_time
        if elapsed_time > 600:
            print(f"⏰ Session expirée ({elapsed_time:.0f}s)")
            generated_response = "Thank you for choosing our airline! I need to assist other passengers now. Have a wonderful trip!"        
            
            audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, character)
            audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
            await asyncio.to_thread(delete_audio_file, audio_file_path)
            
            conversation_history.pop(sarah_key, None)
            conversation_start_time.pop(sarah_key, None)
            active_user_choices.pop(user_id, None)
            
            return {
                "character": "Sarah",
                "user_input": user_input,
                "generated_response": generated_response,
                "audio_base64": audio_base64,
                "user_audio_base64": user_audio_base64,
                "type": "session_end"
            }

        # Si c'est le tout premier message, gérer salutation + réponse
        if is_very_first_message:
            print(f"🎭 Premier message détecté - Génération de la salutation SEULEMENT")
            try:
                # Générer la salutation
                greeting = get_character_greeting('sarah', user_id)
                print(f"🎭 Message de salutation de Sarah: {greeting}")
                
                # Créer la conversation et sauvegarder la salutation
                conversation = create_or_get_conversation(db, user_id, choice)
                
                greeting_audio_path, greeting_duration = await edge_text_to_speech_with_fallback(greeting, character)
                greeting_audio_base64 = await asyncio.to_thread(file_to_base64, greeting_audio_path)
                await asyncio.to_thread(delete_audio_file, greeting_audio_path)
                
                log_message_to_db(
                    db, user_id, conversation.id, greeting,
                    greeting, greeting_audio_base64, ia_audio_duration=greeting_duration
                )
                
                # Ajouter la salutation à l'historique
                conversation_history[sarah_key].append({'input': "", 'response': greeting})
                
                # Pour le premier message, utiliser SEULEMENT la salutation comme réponse
                sarah_history = conversation_history[sarah_key]
                sarah_history.append({'input': user_input, 'response': greeting})
                conversation_history[sarah_key] = sarah_history
                
                # Log la conversation
                log_conversation_and_message(db, user_id, category, user_input, user_input, greeting, user_audio_base64, greeting_audio_base64, ia_audio_duration=greeting_duration)
                print(f"🎯 PREMIER MESSAGE - Retour de la salutation seule: {greeting}")
                
                return {
                    "character": "Sarah",
                    "user_input": user_input,
                    "generated_response": greeting,  # UTILISER LA SALUTATION COMME RÉPONSE UNIQUE
                    "audio_base64": greeting_audio_base64,  # MÊME AUDIO QUE LA SALUTATION
                    "conversation_history": sarah_history,
                    "user_audio_base64": user_audio_base64,
                    "greeting": greeting,
                    "greeting_audio": greeting_audio_base64,
                    "is_first_message": True,
                    "metrics": {
                        "transcription_time": transcription_time,
                    }
                }
            except Exception as greeting_error:
                print(f"❌ Erreur génération salutation: {greeting_error}")
                raise HTTPException(status_code=500, detail=f"Erreur de salutation: {str(greeting_error)}")

        # Messages suivants (conversation normale)
        print(f"💬 Message de conversation normale")
        try:
            ai_function = get_ai_function_for_choice(choice)
            sarah_history = conversation_history.get(sarah_key, [])
            print(f"🤖 Génération réponse Sarah pour: '{user_input}'")
            generated_response = await asyncio.to_thread(ai_function, user_input, sarah_history, user_id)
            print(f"✅ Réponse de Sarah: '{generated_response}'")
            
            # Ajouter à l'historique
            sarah_history.append({'input': user_input, 'response': generated_response})
            conversation_history[sarah_key] = sarah_history
            
            # Générer audio
            audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, character)
            audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
            log_conversation_and_message(db, user_id, category, user_input, user_input, generated_response, user_audio_base64, audio_base64, ia_audio_duration=duration)
            await asyncio.to_thread(delete_audio_file, audio_file_path)
            
            return {
                "character": "Sarah",
                "user_input": user_input,
                "generated_response": generated_response,
                "audio_base64": audio_base64,
                "conversation_history": sarah_history,
                "user_audio_base64": user_audio_base64,
                "is_first_message": False,
                "metrics": {
                    "transcription_time": transcription_time,
                }
            }
        except Exception as response_error:
            print(f"❌ Erreur génération réponse: {response_error}")
            raise HTTPException(status_code=500, detail=f"Erreur de génération de réponse: {str(response_error)}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur générale avec Sarah: {e}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An error occurred during the conversation with Sarah: {str(e)}")

# ===== 🆕 NOUVEAUX ENDPOINTS POUR PERSONNAGES DÉBUTANTS =====

@app.post("/chat_conv/emma")
async def chat_with_emma(
    audio_file: UploadFile = File(None),
    text_message: Optional[str] = Form(None),
    voice: str = "edge_emma",
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_user)
):
    """Conversation avec Emma - Professeure pour DÉBUTANTS ABSOLUS (EDGE-TTS)"""
    try:
        user_id = current_user.id
        choice = "conv_beginner_teacher"
        emma_key = f"{user_id}_emma"
        category = get_category(choice)
        character = "emma"
        print(f"👩‍🏫 Conversation avec Emma pour l'utilisateur {user_id} (Edge-TTS - Débutants)")

        # Traitement AUDIO OU TEXTE
        if text_message and text_message.strip():
            print(f"📝 Message texte reçu: '{text_message}'")
            user_input = text_message.strip()
            user_audio_base64 = None
            transcription_time = 0
        elif audio_file and audio_file.size > 100:
            print(f"🎤 Message audio reçu (taille: {audio_file.size} bytes)")
            try:
                user_input, user_audio_base64, transcription_time = await process_audio_and_transcribe(audio_file)
            except Exception as transcribe_error:
                print(f"❌ Erreur transcription: {transcribe_error}")
                raise HTTPException(status_code=500, detail=f"Erreur de transcription: {str(transcribe_error)}")
        else:
            print(f"❌ Aucun message valide reçu")
            raise HTTPException(status_code=400, detail="Aucun message audio ou texte valide reçu")

        print(f"💬 Message traité: '{user_input}'")

        # Vérifier si c'est le TOUT PREMIER message
        is_very_first_message = emma_key not in conversation_history or len(conversation_history[emma_key]) == 0

        # Initialiser l'historique si nécessaire
        if emma_key not in conversation_history:
            conversation_history[emma_key] = []
            conversation_start_time[emma_key] = time.time()
            active_user_choices[user_id] = choice

        # Vérifier les réponses communes (sauf pour le premier message)
        if not is_very_first_message:
            try:
                common_response = await handle_common_responses(user_input, character, db, user_id, category, user_audio_base64)
                if common_response:
                    common_response["character"] = "Emma"
                    print(f"🔄 Réponse commune utilisée: {common_response['type']}")
                    return common_response
            except Exception as common_error:
                print(f"❌ Erreur réponses communes: {common_error}")

        # Vérifier timeout (10 minutes)
        current_time = time.time()
        start_time = conversation_start_time.get(emma_key, current_time)
        elapsed_time = current_time - start_time
        if elapsed_time > 600:
            print(f"⏰ Session expirée ({elapsed_time:.0f}s)")
            generated_response = "Good job today! You did very well. See you next time!"
            
            audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, character)
            audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
            await asyncio.to_thread(delete_audio_file, audio_file_path)
            
            conversation_history.pop(emma_key, None)
            conversation_start_time.pop(emma_key, None)
            active_user_choices.pop(user_id, None)
            
            return {
                "character": "Emma",
                "user_input": user_input,
                "generated_response": generated_response,
                "audio_base64": audio_base64,
                "user_audio_base64": user_audio_base64,
                "type": "session_end"
            }

        # Si c'est le tout premier message, gérer salutation + réponse
        if is_very_first_message:
            print(f"🎭 Premier message détecté - Génération de la salutation")
            try:
                # Générer la salutation
                greeting = get_character_greeting('emma', user_id)
                print(f"🎭 Message de salutation d'Emma: {greeting}")
                
                # Créer la conversation et sauvegarder la salutation
                conversation = create_or_get_conversation(db, user_id, choice)
                
                greeting_audio_path, greeting_duration = await edge_text_to_speech_with_fallback(greeting, character)
                greeting_audio_base64 = await asyncio.to_thread(file_to_base64, greeting_audio_path)
                await asyncio.to_thread(delete_audio_file, greeting_audio_path)
                
                log_message_to_db(
                    db, user_id, conversation.id, greeting,
                    greeting, greeting_audio_base64, ia_audio_duration=greeting_duration
                )
                
                # Ajouter la salutation à l'historique
                conversation_history[emma_key].append({'input': "", 'response': greeting})
                
                # Générer la réponse
                emma_history = conversation_history[emma_key]
                ai_function = get_ai_function_for_choice(choice)
                print(f"🤖 Génération de la réponse Emma pour: '{user_input}'")
                generated_response = await asyncio.to_thread(ai_function, user_input, emma_history, user_id)
                print(f"✅ Réponse générée par Emma: '{generated_response}'")
                
                # Ajouter à l'historique
                emma_history.append({'input': user_input, 'response': generated_response})
                conversation_history[emma_key] = emma_history
                
                # Générer audio pour la réponse
                audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, character)
                audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
                log_conversation_and_message(db, user_id, category, user_input, user_input, generated_response, user_audio_base64, audio_base64, ia_audio_duration=duration)
                await asyncio.to_thread(delete_audio_file, audio_file_path)
                
                return {
                    "character": "Emma",
                    "user_input": user_input,
                    "generated_response": generated_response,
                    "audio_base64": audio_base64,
                    "conversation_history": emma_history,
                    "user_audio_base64": user_audio_base64,
                    "greeting": greeting,
                    "greeting_audio": greeting_audio_base64,
                    "is_first_message": True,
                    "metrics": {
                        "transcription_time": transcription_time,
                    }
                }
            except Exception as greeting_error:
                print(f"❌ Erreur génération salutation: {greeting_error}")
                raise HTTPException(status_code=500, detail=f"Erreur de salutation: {str(greeting_error)}")

        # Messages suivants (conversation normale)
        print(f"💬 Message de conversation normale")
        try:
            ai_function = get_ai_function_for_choice(choice)
            emma_history = conversation_history.get(emma_key, [])
            print(f"🤖 Génération réponse Emma pour: '{user_input}'")
            generated_response = await asyncio.to_thread(ai_function, user_input, emma_history, user_id)
            print(f"✅ Réponse d'Emma: '{generated_response}'")
            
            # Ajouter à l'historique
            emma_history.append({'input': user_input, 'response': generated_response})
            conversation_history[emma_key] = emma_history
            
            # Générer audio
            audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, character)
            audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
            log_conversation_and_message(db, user_id, category, user_input, user_input, generated_response, user_audio_base64, audio_base64, ia_audio_duration=duration)
            await asyncio.to_thread(delete_audio_file, audio_file_path)
            
            return {
                "character": "Emma",
                "user_input": user_input,
                "generated_response": generated_response,
                "audio_base64": audio_base64,
                "conversation_history": emma_history,
                "user_audio_base64": user_audio_base64,
                "is_first_message": False,
                "metrics": {
                    "transcription_time": transcription_time,
                }
            }
        except Exception as response_error:
            print(f"❌ Erreur génération réponse: {response_error}")
            raise HTTPException(status_code=500, detail=f"Erreur de génération de réponse: {str(response_error)}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur générale avec Emma: {e}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An error occurred during the conversation with Emma: {str(e)}")

@app.post("/chat_conv/tom")
async def chat_with_tom(
    audio_file: UploadFile = File(None),
    text_message: Optional[str] = Form(None),
    voice: str = "edge_tom",
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_user)
):
    """Conversation avec Tom - Serveur restaurant SIMPLE (EDGE-TTS)"""
    try:
        user_id = current_user.id
        choice = "conv_simple_restaurant"
        tom_key = f"{user_id}_tom"
        category = get_category(choice)
        character = "tom"
        print(f"🍕 Conversation avec Tom pour l'utilisateur {user_id} (Edge-TTS - Restaurant)")

        # Traitement AUDIO OU TEXTE
        if text_message and text_message.strip():
            print(f"📝 Message texte reçu: '{text_message}'")
            user_input = text_message.strip()
            user_audio_base64 = None
            transcription_time = 0
        elif audio_file and audio_file.size > 100:
            print(f"🎤 Message audio reçu (taille: {audio_file.size} bytes)")
            try:
                user_input, user_audio_base64, transcription_time = await process_audio_and_transcribe(audio_file)
            except Exception as transcribe_error:
                print(f"❌ Erreur transcription: {transcribe_error}")
                raise HTTPException(status_code=500, detail=f"Erreur de transcription: {str(transcribe_error)}")
        else:
            print(f"❌ Aucun message valide reçu")
            raise HTTPException(status_code=400, detail="Aucun message audio ou texte valide reçu")

        print(f"💬 Message traité: '{user_input}'")

        # Vérifier si c'est le TOUT PREMIER message
        is_very_first_message = tom_key not in conversation_history or len(conversation_history[tom_key]) == 0

        # Initialiser l'historique si nécessaire
        if tom_key not in conversation_history:
            conversation_history[tom_key] = []
            conversation_start_time[tom_key] = time.time()
            active_user_choices[user_id] = choice

        # Vérifier les réponses communes (sauf pour le premier message)
        if not is_very_first_message:
            try:
                common_response = await handle_common_responses(user_input, character, db, user_id, category, user_audio_base64)
                if common_response:
                    common_response["character"] = "Tom"
                    print(f"🔄 Réponse commune utilisée: {common_response['type']}")
                    return common_response
            except Exception as common_error:
                print(f"❌ Erreur réponses communes: {common_error}")

        # Vérifier timeout (10 minutes)
        current_time = time.time()
        start_time = conversation_start_time.get(tom_key, current_time)
        elapsed_time = current_time - start_time
        if elapsed_time > 600:
            print(f"⏰ Session expirée ({elapsed_time:.0f}s)")
            generated_response = "Thank you for coming! Have a good day!"
            
            audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, character)
            audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
            await asyncio.to_thread(delete_audio_file, audio_file_path)
            
            conversation_history.pop(tom_key, None)
            conversation_start_time.pop(tom_key, None)
            active_user_choices.pop(user_id, None)
            
            return {
                "character": "Tom",
                "user_input": user_input,
                "generated_response": generated_response,
                "audio_base64": audio_base64,
                "user_audio_base64": user_audio_base64,
                "type": "session_end"
            }

        # Si c'est le tout premier message, gérer salutation + réponse
        if is_very_first_message:
            print(f"🎭 Premier message détecté - Génération de la salutation")
            try:
                # Générer la salutation
                greeting = get_character_greeting('tom', user_id)
                print(f"🎭 Message de salutation de Tom: {greeting}")
                
                # Créer la conversation et sauvegarder la salutation
                conversation = create_or_get_conversation(db, user_id, choice)
                
                greeting_audio_path, greeting_duration = await edge_text_to_speech_with_fallback(greeting, character)
                greeting_audio_base64 = await asyncio.to_thread(file_to_base64, greeting_audio_path)
                await asyncio.to_thread(delete_audio_file, greeting_audio_path)
                
                log_message_to_db(
                    db, user_id, conversation.id, greeting,
                    greeting, greeting_audio_base64, ia_audio_duration=greeting_duration
                )
                
                # Ajouter la salutation à l'historique
                conversation_history[tom_key].append({'input': "", 'response': greeting})
                
                # Générer la réponse
                tom_history = conversation_history[tom_key]
                ai_function = get_ai_function_for_choice(choice)
                print(f"🤖 Génération de la réponse Tom pour: '{user_input}'")
                generated_response = await asyncio.to_thread(ai_function, user_input, tom_history, user_id)
                print(f"✅ Réponse générée par Tom: '{generated_response}'")
                
                # Ajouter à l'historique
                tom_history.append({'input': user_input, 'response': generated_response})
                conversation_history[tom_key] = tom_history
                
                # Générer audio pour la réponse
                audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, character)
                audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
                log_conversation_and_message(db, user_id, category, user_input, user_input, generated_response, user_audio_base64, audio_base64, ia_audio_duration=duration)
                await asyncio.to_thread(delete_audio_file, audio_file_path)
                
                return {
                    "character": "Tom",
                    "user_input": user_input,
                    "generated_response": generated_response,
                    "audio_base64": audio_base64,
                    "conversation_history": tom_history,
                    "user_audio_base64": user_audio_base64,
                    "greeting": greeting,
                    "greeting_audio": greeting_audio_base64,
                    "is_first_message": True,
                    "metrics": {
                        "transcription_time": transcription_time,
                    }
                }
            except Exception as greeting_error:
                print(f"❌ Erreur génération salutation: {greeting_error}")
                raise HTTPException(status_code=500, detail=f"Erreur de salutation: {str(greeting_error)}")

        # Messages suivants (conversation normale)
        print(f"💬 Message de conversation normale")
        try:
            ai_function = get_ai_function_for_choice(choice)
            tom_history = conversation_history.get(tom_key, [])
            print(f"🤖 Génération réponse Tom pour: '{user_input}'")
            generated_response = await asyncio.to_thread(ai_function, user_input, tom_history, user_id)
            print(f"✅ Réponse de Tom: '{generated_response}'")
            
            # Ajouter à l'historique
            tom_history.append({'input': user_input, 'response': generated_response})
            conversation_history[tom_key] = tom_history
            
            # Générer audio
            audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, character)
            audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
            log_conversation_and_message(db, user_id, category, user_input, user_input, generated_response, user_audio_base64, audio_base64, ia_audio_duration=duration)
            await asyncio.to_thread(delete_audio_file, audio_file_path)
            
            return {
                "character": "Tom",
                "user_input": user_input,
                "generated_response": generated_response,
                "audio_base64": audio_base64,
                "conversation_history": tom_history,
                "user_audio_base64": user_audio_base64,
                "is_first_message": False,
                "metrics": {
                    "transcription_time": transcription_time,
                }
            }
        except Exception as response_error:
            print(f"❌ Erreur génération réponse: {response_error}")
            raise HTTPException(status_code=500, detail=f"Erreur de génération de réponse: {str(response_error)}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur générale avec Tom: {e}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An error occurred during the conversation with Tom: {str(e)}")

@app.post("/chat_conv/lucy")
async def chat_with_lucy(
    audio_file: UploadFile = File(None),
    text_message: Optional[str] = Form(None),
    voice: str = "edge_lucy",
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_user)
):
    """Conversation avec Lucy - Vendeuse SIMPLE (EDGE-TTS)"""
    try:
        user_id = current_user.id
        choice = "conv_simple_shopping"
        lucy_key = f"{user_id}_lucy"
        category = get_category(choice)
        character = "lucy"
        print(f"🛍️ Conversation avec Lucy pour l'utilisateur {user_id} (Edge-TTS - Shopping)")

        # Traitement AUDIO OU TEXTE
        if text_message and text_message.strip():
            print(f"📝 Message texte reçu: '{text_message}'")
            user_input = text_message.strip()
            user_audio_base64 = None
            transcription_time = 0
        elif audio_file and audio_file.size > 100:
            print(f"🎤 Message audio reçu (taille: {audio_file.size} bytes)")
            try:
                user_input, user_audio_base64, transcription_time = await process_audio_and_transcribe(audio_file)
            except Exception as transcribe_error:
                print(f"❌ Erreur transcription: {transcribe_error}")
                raise HTTPException(status_code=500, detail=f"Erreur de transcription: {str(transcribe_error)}")
        else:
            print(f"❌ Aucun message valide reçu")
            raise HTTPException(status_code=400, detail="Aucun message audio ou texte valide reçu")

        print(f"💬 Message traité: '{user_input}'")

        # Vérifier si c'est le TOUT PREMIER message
        is_very_first_message = lucy_key not in conversation_history or len(conversation_history[lucy_key]) == 0

        # Initialiser l'historique si nécessaire
        if lucy_key not in conversation_history:
            conversation_history[lucy_key] = []
            conversation_start_time[lucy_key] = time.time()
            active_user_choices[user_id] = choice

        # Vérifier les réponses communes (sauf pour le premier message)
        if not is_very_first_message:
            try:
                common_response = await handle_common_responses(user_input, character, db, user_id, category, user_audio_base64)
                if common_response:
                    common_response["character"] = "Lucy"
                    print(f"🔄 Réponse commune utilisée: {common_response['type']}")
                    return common_response
            except Exception as common_error:
                print(f"❌ Erreur réponses communes: {common_error}")

        # Vérifier timeout (10 minutes)
        current_time = time.time()
        start_time = conversation_start_time.get(lucy_key, current_time)
        elapsed_time = current_time - start_time
        if elapsed_time > 600:
            print(f"⏰ Session expirée ({elapsed_time:.0f}s)")
            generated_response = "Thank you for shopping with us! Have a nice day!"
            
            audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, character)
            audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
            await asyncio.to_thread(delete_audio_file, audio_file_path)
            
            conversation_history.pop(lucy_key, None)
            conversation_start_time.pop(lucy_key, None)
            active_user_choices.pop(user_id, None)
            
            return {
                "character": "Lucy",
                "user_input": user_input,
                "generated_response": generated_response,
                "audio_base64": audio_base64,
                "user_audio_base64": user_audio_base64,
                "type": "session_end"
            }

        # Si c'est le tout premier message, gérer salutation + réponse
        if is_very_first_message:
            print(f"🎭 Premier message détecté - Génération de la salutation")
            try:
                # Générer la salutation
                greeting = get_character_greeting('lucy', user_id)
                print(f"🎭 Message de salutation de Lucy: {greeting}")
                
                # Créer la conversation et sauvegarder la salutation
                conversation = create_or_get_conversation(db, user_id, choice)
                
                greeting_audio_path, greeting_duration = await edge_text_to_speech_with_fallback(greeting, character)
                greeting_audio_base64 = await asyncio.to_thread(file_to_base64, greeting_audio_path)
                await asyncio.to_thread(delete_audio_file, greeting_audio_path)
                
                log_message_to_db(
                    db, user_id, conversation.id, greeting,
                    greeting, greeting_audio_base64, ia_audio_duration=greeting_duration
                )
                
                # Ajouter la salutation à l'historique
                conversation_history[lucy_key].append({'input': "", 'response': greeting})
                
                # Générer la réponse
                lucy_history = conversation_history[lucy_key]
                ai_function = get_ai_function_for_choice(choice)
                print(f"🤖 Génération de la réponse Lucy pour: '{user_input}'")
                generated_response = await asyncio.to_thread(ai_function, user_input, lucy_history, user_id)
                print(f"✅ Réponse générée par Lucy: '{generated_response}'")
                
                # Ajouter à l'historique
                lucy_history.append({'input': user_input, 'response': generated_response})
                conversation_history[lucy_key] = lucy_history
                
                # Générer audio pour la réponse
                audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, character)
                audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
                log_conversation_and_message(db, user_id, category, user_input, user_input, generated_response, user_audio_base64, audio_base64, ia_audio_duration=duration)
                await asyncio.to_thread(delete_audio_file, audio_file_path)
                
                return {
                    "character": "Lucy",
                    "user_input": user_input,
                    "generated_response": generated_response,
                    "audio_base64": audio_base64,
                    "conversation_history": lucy_history,
                    "user_audio_base64": user_audio_base64,
                    "greeting": greeting,
                    "greeting_audio": greeting_audio_base64,
                    "is_first_message": True,
                    "metrics": {
                        "transcription_time": transcription_time,
                    }
                }
            except Exception as greeting_error:
                print(f"❌ Erreur génération salutation: {greeting_error}")
                raise HTTPException(status_code=500, detail=f"Erreur de salutation: {str(greeting_error)}")

        # Messages suivants (conversation normale)
        print(f"💬 Message de conversation normale")
        try:
            ai_function = get_ai_function_for_choice(choice)
            lucy_history = conversation_history.get(lucy_key, [])
            print(f"🤖 Génération réponse Lucy pour: '{user_input}'")
            generated_response = await asyncio.to_thread(ai_function, user_input, lucy_history, user_id)
            print(f"✅ Réponse de Lucy: '{generated_response}'")
            
            # Ajouter à l'historique
            lucy_history.append({'input': user_input, 'response': generated_response})
            conversation_history[lucy_key] = lucy_history
            
            # Générer audio
            audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, character)
            audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
            log_conversation_and_message(db, user_id, category, user_input, user_input, generated_response, user_audio_base64, audio_base64, ia_audio_duration=duration)
            await asyncio.to_thread(delete_audio_file, audio_file_path)
            
            return {
                "character": "Lucy",
                "user_input": user_input,
                "generated_response": generated_response,
                "audio_base64": audio_base64,
                "conversation_history": lucy_history,
                "user_audio_base64": user_audio_base64,
                "is_first_message": False,
                "metrics": {
                    "transcription_time": transcription_time,
                }
            }
        except Exception as response_error:
            print(f"❌ Erreur génération réponse: {response_error}")
            raise HTTPException(status_code=500, detail=f"Erreur de génération de réponse: {str(response_error)}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur générale avec Lucy: {e}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An error occurred during the conversation with Lucy: {str(e)}")

@app.post("/chat_conv/ben")
async def chat_with_ben(
    audio_file: UploadFile = File(None),
    text_message: Optional[str] = Form(None),
    voice: str = "edge_ben",
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_user)
):
    """Conversation avec Ben - Guide touristique SIMPLE (EDGE-TTS)"""
    try:
        user_id = current_user.id
        choice = "conv_simple_directions"
        ben_key = f"{user_id}_ben"
        category = get_category(choice)
        character = "ben"
        print(f"🗺️ Conversation avec Ben pour l'utilisateur {user_id} (Edge-TTS - Directions)")

        # Traitement AUDIO OU TEXTE
        if text_message and text_message.strip():
            print(f"📝 Message texte reçu: '{text_message}'")
            user_input = text_message.strip()
            user_audio_base64 = None
            transcription_time = 0
        elif audio_file and audio_file.size > 100:
            print(f"🎤 Message audio reçu (taille: {audio_file.size} bytes)")
            try:
                user_input, user_audio_base64, transcription_time = await process_audio_and_transcribe(audio_file)
            except Exception as transcribe_error:
                print(f"❌ Erreur transcription: {transcribe_error}")
                raise HTTPException(status_code=500, detail=f"Erreur de transcription: {str(transcribe_error)}")
        else:
            print(f"❌ Aucun message valide reçu")
            raise HTTPException(status_code=400, detail="Aucun message audio ou texte valide reçu")

        print(f"💬 Message traité: '{user_input}'")

        # Vérifier si c'est le TOUT PREMIER message
        is_very_first_message = ben_key not in conversation_history or len(conversation_history[ben_key]) == 0

        # Initialiser l'historique si nécessaire
        if ben_key not in conversation_history:
            conversation_history[ben_key] = []
            conversation_start_time[ben_key] = time.time()
            active_user_choices[user_id] = choice

        # Vérifier les réponses communes (sauf pour le premier message)
        if not is_very_first_message:
            try:
                common_response = await handle_common_responses(user_input, character, db, user_id, category, user_audio_base64)
                if common_response:
                    common_response["character"] = "Ben"
                    print(f"🔄 Réponse commune utilisée: {common_response['type']}")
                    return common_response
            except Exception as common_error:
                print(f"❌ Erreur réponses communes: {common_error}")

        # Vérifier timeout (10 minutes)
        current_time = time.time()
        start_time = conversation_start_time.get(ben_key, current_time)
        elapsed_time = current_time - start_time
        if elapsed_time > 600:
            print(f"⏰ Session expirée ({elapsed_time:.0f}s)")
            generated_response = "I hope you find your way! Have a wonderful trip!"
            
            audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, character)
            audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
            await asyncio.to_thread(delete_audio_file, audio_file_path)
            
            conversation_history.pop(ben_key, None)
            conversation_start_time.pop(ben_key, None)
            active_user_choices.pop(user_id, None)
            
            return {
                "character": "Ben",
                "user_input": user_input,
                "generated_response": generated_response,
                "audio_base64": audio_base64,
                "user_audio_base64": user_audio_base64,
                "type": "session_end"
            }

        # Si c'est le tout premier message, gérer salutation + réponse
        if is_very_first_message:
            print(f"🎭 Premier message détecté - Génération de la salutation")
            try:
                # Générer la salutation
                greeting = get_character_greeting('ben', user_id)
                print(f"🎭 Message de salutation de Ben: {greeting}")
                
                # Créer la conversation et sauvegarder la salutation
                conversation = create_or_get_conversation(db, user_id, choice)
                
                greeting_audio_path, greeting_duration = await edge_text_to_speech_with_fallback(greeting, character)
                greeting_audio_base64 = await asyncio.to_thread(file_to_base64, greeting_audio_path)
                await asyncio.to_thread(delete_audio_file, greeting_audio_path)
                
                log_message_to_db(
                    db, user_id, conversation.id, greeting,
                    greeting, greeting_audio_base64, ia_audio_duration=greeting_duration
                )
                
                # Ajouter la salutation à l'historique
                conversation_history[ben_key].append({'input': "", 'response': greeting})
                
                # Générer la réponse
                ben_history = conversation_history[ben_key]
                ai_function = get_ai_function_for_choice(choice)
                print(f"🤖 Génération de la réponse Ben pour: '{user_input}'")
                generated_response = await asyncio.to_thread(ai_function, user_input, ben_history, user_id)
                print(f"✅ Réponse générée par Ben: '{generated_response}'")
                
                # Ajouter à l'historique
                ben_history.append({'input': user_input, 'response': generated_response})
                conversation_history[ben_key] = ben_history
                
                # Générer audio pour la réponse
                audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, character)
                audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
                log_conversation_and_message(db, user_id, category, user_input, user_input, generated_response, user_audio_base64, audio_base64, ia_audio_duration=duration)
                await asyncio.to_thread(delete_audio_file, audio_file_path)
                
                return {
                    "character": "Ben",
                    "user_input": user_input,
                    "generated_response": generated_response,
                    "audio_base64": audio_base64,
                    "conversation_history": ben_history,
                    "user_audio_base64": user_audio_base64,
                    "greeting": greeting,
                    "greeting_audio": greeting_audio_base64,
                    "is_first_message": True,
                    "metrics": {
                        "transcription_time": transcription_time,
                    }
                }
            except Exception as greeting_error:
                print(f"❌ Erreur génération salutation: {greeting_error}")
                raise HTTPException(status_code=500, detail=f"Erreur de salutation: {str(greeting_error)}")

        # Messages suivants (conversation normale)
        print(f"💬 Message de conversation normale")
        try:
            ai_function = get_ai_function_for_choice(choice)
            ben_history = conversation_history.get(ben_key, [])
            print(f"🤖 Génération réponse Ben pour: '{user_input}'")
            generated_response = await asyncio.to_thread(ai_function, user_input, ben_history, user_id)
            print(f"✅ Réponse de Ben: '{generated_response}'")
            
            # Ajouter à l'historique
            ben_history.append({'input': user_input, 'response': generated_response})
            conversation_history[ben_key] = ben_history
            
            # Générer audio
            audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, character)
            audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
            log_conversation_and_message(db, user_id, category, user_input, user_input, generated_response, user_audio_base64, audio_base64, ia_audio_duration=duration)
            await asyncio.to_thread(delete_audio_file, audio_file_path)
            
            return {
                "character": "Ben",
                "user_input": user_input,
                "generated_response": generated_response,
                "audio_base64": audio_base64,
                "conversation_history": ben_history,
                "user_audio_base64": user_audio_base64,
                "is_first_message": False,
                "metrics": {
                    "transcription_time": transcription_time,
                }
            }
        except Exception as response_error:
            print(f"❌ Erreur génération réponse: {response_error}")
            raise HTTPException(status_code=500, detail=f"Erreur de génération de réponse: {str(response_error)}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur générale avec Ben: {e}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An error occurred during the conversation with Ben: {str(e)}")

# ===== ENDPOINT GÉNÉRIQUE MISE À JOUR =====

@app.post("/chat_conv")
async def chat_with_brain(
    choice: str = Form(...),
    audio_file: UploadFile = File(None),
    text_message: Optional[str] = Form(None),
    voice: str = "edge_alice",  # 🔥 EDGE-TTS PAR DÉFAUT
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_user)
):
    """Endpoint générique pour chat - REDIRECTION VERS LES PERSONNAGES SPÉCIFIQUES (EDGE-TTS) - AVEC NOUVEAUX PERSONNAGES"""
    user_id = current_user.id
    print(f"🎯 Chat générique - User: {user_id}, Choice: {choice}")
    
    # Adapter la voix selon le choix si pas spécifié
    if voice == "edge_alice":  # Voix par défaut
        character_voice_mapping = {
            "conv_greetings_common_conversations": "edge_alice",
            "conv_taxi_booking": "edge_mike",
            "conv_airport_ticket": "edge_sarah",
            # 🆕 NOUVEAUX PERSONNAGES
            "conv_beginner_teacher": "edge_emma",
            "conv_simple_restaurant": "edge_tom",
            "conv_simple_shopping": "edge_lucy",
            "conv_simple_directions": "edge_ben"
        }
        voice = character_voice_mapping.get(choice, "edge_alice")
    
    # Rediriger vers l'endpoint spécifique selon le choix
    try:
        if choice == "conv_greetings_common_conversations":
            print("🌟 Redirection vers Alice (Edge-TTS)")
            response = await chat_with_alice(audio_file, text_message, voice, db, current_user)
        elif choice == "conv_taxi_booking":
            print("🚕 Redirection vers Mike (Edge-TTS)")
            response = await chat_with_mike(audio_file, text_message, voice, db, current_user)
        elif choice == "conv_airport_ticket":
            print("✈️ Redirection vers Sarah (Edge-TTS)")
            response = await chat_with_sarah(audio_file, text_message, voice, db, current_user)
        # 🆕 NOUVEAUX PERSONNAGES DÉBUTANTS
        elif choice == "conv_beginner_teacher":
            print("👩‍🏫 Redirection vers Emma (Edge-TTS)")
            response = await chat_with_emma(audio_file, text_message, voice, db, current_user)
        elif choice == "conv_simple_restaurant":
            print("🍕 Redirection vers Tom (Edge-TTS)")
            response = await chat_with_tom(audio_file, text_message, voice, db, current_user)
        elif choice == "conv_simple_shopping":
            print("🛍️ Redirection vers Lucy (Edge-TTS)")
            response = await chat_with_lucy(audio_file, text_message, voice, db, current_user)
        elif choice == "conv_simple_directions":
            print("🗺️ Redirection vers Ben (Edge-TTS)")
            response = await chat_with_ben(audio_file, text_message, voice, db, current_user)
        else:
            print(f"❌ Choix non reconnu: {choice}")
            raise HTTPException(status_code=400, detail=f"Invalid choice parameter: {choice}")
        
        # Ajouter les informations du choix à la réponse
        character = CHOICE_TO_CHARACTER.get(choice)
        response["choice"] = choice
        response["character_info"] = CHARACTER_INFO.get(character, {})
        return response
        
    except Exception as e:
        print(f"❌ Erreur lors de la redirection: {e}")
        raise e

# ===== ROUTES D'ANALYSE =====
@app.get("/analyze_session/{conversation_id}", response_model=AnalysisResponse)
def analyze_session(conversation_id: int, db: Session = Depends(get_db), current_user: DBUser = Depends(get_current_user)):
    try:
        # 🔥 CORRECTION : Vérifier que la conversation appartient à l'utilisateur
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user1_id == current_user.id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found or access denied")
        
        # Récupérer les messages avec gestion d'erreur
        try:
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
            
            all_messages = db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).all()
            
            # Calculer les phrases correctes (exclure les marqueurs d'erreur et traductions)
            excluded_markers = ["unclear_responses", "french_responses", "translations"]
            correct_phrases_count = len([
                msg for msg in all_messages 
                if not msg.marker or msg.marker not in excluded_markers
            ])
            
            # 🔥 CORRECTION : Gestion des dates avec timezone
            try:
                if conversation.end_time and conversation.start_time:
                    # S'assurer que les deux dates ont le même type de timezone
                    start_time = conversation.start_time
                    end_time = conversation.end_time
                    
                    # Si une date a timezone et l'autre non, normaliser
                    if start_time.tzinfo is None and end_time.tzinfo is not None:
                        start_time = start_time.replace(tzinfo=timezone.utc)
                    elif start_time.tzinfo is not None and end_time.tzinfo is None:
                        end_time = end_time.replace(tzinfo=timezone.utc)
                    
                    duration = end_time - start_time
                else:
                    # Utiliser le temps actuel si pas de end_time
                    start_time = conversation.start_time
                    current_time = datetime.now(timezone.utc)
                    
                    # Normaliser la timezone
                    if start_time.tzinfo is None:
                        start_time = start_time.replace(tzinfo=timezone.utc)
                    
                    duration = current_time - start_time
                
                # Formatage sécurisé de la durée
                total_seconds = int(duration.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                
                if hours > 0:
                    duration_str = f"{hours}h {minutes}m"
                elif minutes > 0:
                    duration_str = f"{minutes}m"
                else:
                    duration_str = "< 1m"
                    
            except Exception as duration_error:
                print(f"⚠️ Erreur calcul durée: {duration_error}")
                duration_str = "N/A"
            
            total_messages = len(all_messages)
            
            # Calcul du temps de réponse moyen avec gestion d'erreur améliorée
            try:
                avg_user_response_time = calculate_avg_user_response_time_safe(all_messages)
            except Exception as time_error:
                print(f"⚠️ Erreur calcul temps réponse: {time_error}")
                avg_user_response_time = "N/A"
            
            # Construction de la réponse avec valeurs par défaut
            analysis_result = {
                "translations": [
                    (msg.user_input or "N/A", msg.response or "N/A") 
                    for msg in translations
                ],
                "unclear_responses": [
                    {
                        "response": msg.content or "N/A", 
                        "suggestion": msg.suggestion or "Aucune suggestion"
                    } for msg in unclear_responses
                ],
                "french_responses": [msg.content or "N/A" for msg in french_responses],
                "french_responses_count": len(french_responses),
                "unclear_responses_count": len(unclear_responses),
                "translations_count": len(translations),
                "correct_phrases_count": max(0, correct_phrases_count),
                "duration": duration_str,
                "category": conversation.category or "unknown",
                "total_messages": total_messages,
                "avg_user_response_time": avg_user_response_time
            }
            
            print(f"✅ Analyse session {conversation_id} réussie: {total_messages} messages")
            return analysis_result
            
        except Exception as messages_error:
            print(f"❌ Erreur traitement messages session {conversation_id}: {messages_error}")
            # Retourner une analyse vide plutôt qu'une erreur
            return {
                "translations": [],
                "unclear_responses": [],
                "french_responses": [],
                "french_responses_count": 0,
                "unclear_responses_count": 0,
                "translations_count": 0,
                "correct_phrases_count": 0,
                "duration": "N/A",
                "category": conversation.category or "unknown",
                "total_messages": 0,
                "avg_user_response_time": "N/A"
            }
            
    except HTTPException:
        # Re-lancer les erreurs HTTP
        raise
    except Exception as e:
        print(f"❌ Erreur générale analyse session {conversation_id}: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors de l'analyse de la session: {str(e)}"
        )
    
def calculate_avg_user_response_time_safe(messages):
    """Calcule le temps de réponse moyen avec gestion complète des timezones"""
    try:
        if not messages or len(messages) < 2:
            return "N/A"
        
        # Trier les messages par timestamp
        sorted_messages = sorted(messages, key=lambda msg: msg.timestamp if msg.timestamp else datetime.min.replace(tzinfo=timezone.utc))
        
        time_differences = []
        for i in range(1, len(sorted_messages)):
            try:
                current_msg = sorted_messages[i]
                previous_msg = sorted_messages[i-1]
                
                # Vérifier que les messages ont des timestamps valides
                if not current_msg.timestamp or not previous_msg.timestamp:
                    continue
                
                # Normaliser les timezones
                current_time = current_msg.timestamp
                previous_time = previous_msg.timestamp
                
                # S'assurer que les deux ont le même type de timezone
                if current_time.tzinfo is None and previous_time.tzinfo is not None:
                    current_time = current_time.replace(tzinfo=timezone.utc)
                elif current_time.tzinfo is not None and previous_time.tzinfo is None:
                    previous_time = previous_time.replace(tzinfo=timezone.utc)
                elif current_time.tzinfo is None and previous_time.tzinfo is None:
                    # Les deux sont naive, les traiter comme UTC
                    current_time = current_time.replace(tzinfo=timezone.utc)
                    previous_time = previous_time.replace(tzinfo=timezone.utc)
                
                # Calculer la différence de temps
                time_diff = (current_time - previous_time).total_seconds()
                
                # Soustraire la durée audio de l'IA si disponible
                if hasattr(previous_msg, 'ia_audio_duration') and previous_msg.ia_audio_duration:
                    time_diff -= previous_msg.ia_audio_duration
                
                # Ignorer les différences négatives ou trop grandes (> 10 minutes)
                if 0 < time_diff < 600:
                    time_differences.append(time_diff)
                    
            except Exception as e:
                print(f"⚠️ Erreur calcul temps réponse message {i}: {e}")
                continue
        
        if not time_differences:
            return "N/A"
        
        avg_response_time = sum(time_differences) / len(time_differences)
        
        # Formatage du résultat
        if avg_response_time >= 60:
            minutes = int(avg_response_time // 60)
            seconds = int(avg_response_time % 60)
            return f"{minutes}m {seconds}s"
        else:
            seconds = int(avg_response_time)
            return f"{seconds}s"
            
    except Exception as e:
        print(f"❌ Erreur globale calcul temps réponse: {e}")
        return "N/A"

@app.post("/get_suggestions")
def get_suggestions(request: dict, db: Session = Depends(get_db), current_user: DBUser = Depends(get_current_user)):
    unclear_response = request.get('unclear_response', '')
    if unclear_response:
        prompt = f"Improve the following unclear response in one sentence easy: {unclear_response}"
        suggestion = generate_phi3_response(prompt)
        return {"suggestion": suggestion}
    else:
        raise HTTPException(status_code=400, detail="Unclear response not provided.")

# ===== ROUTES UTILITAIRES =====
@app.post("/chat_conv/stop")
async def stop_chat(current_user: DBUser = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        user_id = current_user.id
        # Nettoyer le choix actif de l'utilisateur
        active_user_choices.pop(user_id, None)
        
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
def get_user_conversations(current_user: DBUser = Depends(get_current_user), db: Session = Depends(get_db)):
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
        # IMPORTANT: Utiliser Edge-TTS pour les messages texte génériques aussi
        prompt = request.message
        voice = request.voice if hasattr(request, 'voice') and request.voice else "alice"  # 🔥 Par défaut alice
        print(f"🎙️ Utilisation de Edge-TTS pour message générique : {voice}")
        print(f"📝 Message texte générique (Phi3): {prompt}")
        
        generated_response = generate_phi3_response(prompt)
        
        # 🔥 UTILISER EDGE-TTS AU LIEU DE L'ANCIEN SYSTÈME
        audio_file_path, duration = await edge_text_to_speech_with_fallback(generated_response, voice)
        audio_base64 = await asyncio.to_thread(file_to_base64, audio_file_path)
        await asyncio.to_thread(delete_audio_file, audio_file_path)
        
        return {
            "generated_response": generated_response,
            "audio_base64": audio_base64
        }
    except Exception as e:
        print(f"❌ Erreur dans send_message : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/r_a_m/{conversation_id}/messages", response_model=List[MessageSchema])
def get_conversation_messages(
    conversation_id: int,
    category: Optional[str] = None,
    current_user: DBUser = Depends(get_current_user),
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

# ===== 🆕 ENDPOINTS SPÉCIAUX POUR DÉBUTANTS =====

@app.post("/test_beginner_characters")
async def test_beginner_characters():
    """Test rapide de tous les nouveaux personnages débutants"""
    test_message = "Hello"
    results = {}
    
    characters = ["emma", "tom", "lucy", "ben"]
    
    for character in characters:
        try:
            print(f"🧪 Test {character}...")
            
            # Test de génération de salutation
            greeting = get_character_greeting(character, 12345)
            
            # Test de réponse fallback
            from back.prompt.conversation_personalities import get_simple_beginner_fallback
            fallback = get_simple_beginner_fallback(test_message, character)
            
            # Test de validation de simplicité
            from back.prompt.conversation_personalities import is_simple_enough_for_beginners
            is_simple = is_simple_enough_for_beginners(greeting)
            
            results[character] = {
                "success": True,
                "greeting": greeting,
                "fallback": fallback,
                "is_simple": is_simple,
                "greeting_length": len(greeting.split()),
                "fallback_length": len(fallback.split())
            }
            
        except Exception as e:
            results[character] = {
                "success": False,
                "error": str(e)
            }
    
    return {
        "test_message": test_message,
        "results": results,
        "overall_success": all(r.get("success", False) for r in results.values())
    }

@app.get("/beginner_characters_info")
async def get_beginner_characters_info():
    """Informations sur les nouveaux personnages débutants"""
    return {
        "beginner_characters": {
            "emma": {
                "name": "Emma",
                "role": "English Teacher for Beginners",
                "speciality": "Very simple vocabulary, encouraging, patient",
                "example_phrases": [
                    "Hello! I am Emma.",
                    "Very good!",
                    "What is your name?",
                    "Do you like coffee?"
                ]
            },
            "tom": {
                "name": "Tom", 
                "role": "Restaurant Waiter",
                "speciality": "Basic food vocabulary, simple orders",
                "example_phrases": [
                    "What do you want to eat?",
                    "Pizza or burger?",
                    "That is 10 dollars.",
                    "Here is your food."
                ]
            },
            "lucy": {
                "name": "Lucy",
                "role": "Shop Assistant", 
                "speciality": "Clothing vocabulary, colors, sizes",
                "example_phrases": [
                    "What size do you need?",
                    "This shirt is nice.",
                    "Do you like this color?",
                    "The changing room is there."
                ]
            },
            "ben": {
                "name": "Ben",
                "role": "Tourist Guide",
                "speciality": "Simple directions, basic locations",
                "example_phrases": [
                    "Go straight and turn left.",
                    "It is very close.",
                    "Walk 5 minutes.",
                    "Do you see the big building?"
                ]
            }
        },
        "learning_progression": {
            "level_1": "Emma (Basic teacher - start here)",
            "level_2": "Tom (Restaurant ordering)",
            "level_3": "Lucy (Shopping basics)",
            "level_4": "Ben (Asking directions)",
            "level_5": "Alice (General conversation)",
            "level_6": "Mike & Sarah (Professional scenarios)"
        },
        "beginner_features": [
            "Maximum 10 words per sentence",
            "Only basic vocabulary (1-6 letter words)",
            "Lots of encouragement and patience",
            "Simple yes/no questions",
            "Repetition of important words",
            "No complex grammar",
            "Real-world practical situations"
        ]
    }

@app.get("/beginner_stats/{user_id}")
async def get_beginner_learning_stats(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_user)
):
    """Statistiques d'apprentissage spécifiques aux débutants"""
    
    # Vérifier que l'utilisateur peut accéder à ces stats
    if current_user.id != user_id and current_user.role != "ADMIN":
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Récupérer les conversations des personnages débutants
    beginner_categories = [
        "conv_beginner_teacher",
        "conv_simple_restaurant", 
        "conv_simple_shopping",
        "conv_simple_directions"
    ]
    
    stats = {}
    total_conversations = 0
    total_messages = 0
    
    for category in beginner_categories:
        conversations = db.query(Conversation).filter(
            Conversation.user1_id == user_id,
            Conversation.category == category
        ).all()
        
        category_messages = 0
        for conv in conversations:
            messages = db.query(Message).filter(Message.conversation_id == conv.id).count()
            category_messages += messages
        
        character_name = CHOICE_TO_CHARACTER.get(category, "unknown")
        character_info = CHARACTER_INFO.get(character_name, {})
        
        stats[category] = {
            "character": character_info.get("name", "Unknown"),
            "role": character_info.get("role", "Unknown"),
            "conversation_count": len(conversations),
            "message_count": category_messages,
            "last_conversation": conversations[-1].start_time.isoformat() if conversations else None
        }
        
        total_conversations += len(conversations)
        total_messages += category_messages
    
    # Calculer la progression
    progression_score = 0
    if stats["conv_beginner_teacher"]["conversation_count"] > 0:
        progression_score += 25
    if stats["conv_simple_restaurant"]["conversation_count"] > 0:
        progression_score += 25  
    if stats["conv_simple_shopping"]["conversation_count"] > 0:
        progression_score += 25
    if stats["conv_simple_directions"]["conversation_count"] > 0:
        progression_score += 25
    
    # Recommandations
    def get_next_recommended_character(stats):
        """Recommande le prochain personnage à essayer"""
        
        if stats["conv_beginner_teacher"]["conversation_count"] == 0:
            return {
                "character": "Emma",
                "reason": "Start with the basics - Emma will teach you simple English",
                "endpoint": "/chat_conv/emma"
            }
        elif stats["conv_simple_restaurant"]["conversation_count"] < 3:
            return {
                "character": "Tom", 
                "reason": "Practice ordering food - very useful for daily life",
                "endpoint": "/chat_conv/tom"
            }
        elif stats["conv_simple_shopping"]["conversation_count"] < 3:
            return {
                "character": "Lucy",
                "reason": "Learn shopping vocabulary - clothes, colors, sizes", 
                "endpoint": "/chat_conv/lucy"
            }
        elif stats["conv_simple_directions"]["conversation_count"] < 3:
            return {
                "character": "Ben",
                "reason": "Master asking for directions - essential for traveling",
                "endpoint": "/chat_conv/ben"
            }
        else:
            return {
                "character": "Alice",
                "reason": "You're ready for general conversation practice!",
                "endpoint": "/chat_conv/alice"
            }
    
    recommendations = []
    if stats["conv_beginner_teacher"]["conversation_count"] == 0:
        recommendations.append("Start with Emma the teacher to learn basic English")
    elif stats["conv_simple_restaurant"]["conversation_count"] == 0:
        recommendations.append("Try ordering food with Tom the waiter")
    elif stats["conv_simple_shopping"]["conversation_count"] == 0:
        recommendations.append("Practice shopping with Lucy")
    elif stats["conv_simple_directions"]["conversation_count"] == 0:
        recommendations.append("Learn to ask directions with Ben")
    else:
        recommendations.append("Great job! You can now try Alice for general conversation")
    
    return {
        "user_id": user_id,
        "beginner_stats": stats,
        "summary": {
            "total_beginner_conversations": total_conversations,
            "total_beginner_messages": total_messages,
            "progression_percentage": progression_score,
            "level": "Beginner" if progression_score < 100 else "Intermediate Ready"
        },
        "recommendations": recommendations,
        "next_character": get_next_recommended_character(stats)
    }

# 🔧 ENDPOINTS DE TEST EDGE-TTS
@app.post("/test_edge_tts")
async def test_edge_tts_endpoint(
    text: str = "Hello, this is a test of Edge-TTS",
    character: str = "alice"
):
    """Endpoint de test pour Edge-TTS"""
    try:
        print(f"🧪 Test Edge-TTS: '{text}' avec {character}")
        audio_path, duration = await edge_text_to_speech_with_fallback(text, character)
        audio_base64 = await asyncio.to_thread(file_to_base64, audio_path)
        # Nettoyer le fichier de test
        await asyncio.to_thread(delete_audio_file, audio_path)
        return {
            "success": True,
            "text": text,
            "character": character,
            "duration": duration,
            "audio_base64": audio_base64,
            "message": "Edge-TTS test successful"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Edge-TTS test failed"
        }

@app.get("/edge_tts_info")
async def get_edge_tts_info():
    """Informations sur Edge-TTS"""
    try:
        from back.edge_tts import get_available_edge_voices, EDGE_VOICES
        # Obtenir les voix disponibles
        available_voices = await get_available_edge_voices()
        return {
            "status": "operational",
            "configured_voices": EDGE_VOICES,
            "character_mapping": DEFAULT_EDGE_VOICES,
            "available_voices_count": len(available_voices),
            "sample_voices": available_voices[:10] if available_voices else []
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/tts/voices")
async def get_available_tts_voices():
    """Lister toutes les voix TTS disponibles"""
    try:
        from back.edge_tts import get_available_edge_voices, EDGE_VOICES, EDGE_FALLBACK_VOICES
        # Voix Edge disponibles
        edge_voices = await get_available_edge_voices()
        return {
            "edge_tts": {
                "configured_characters": EDGE_VOICES,
                "fallback_voices": EDGE_FALLBACK_VOICES,
                "available_count": len(edge_voices),
                "sample_voices": edge_voices[:20] if edge_voices else []
            },
            "character_mapping": DEFAULT_EDGE_VOICES,
            "recommended": {
                "alice": "en-US-JennyNeural (friendly female)",
                "mike": "en-US-BrianNeural (professional male)",
                "sarah": "en-US-AriaNeural (customer service)",
                # 🆕 NOUVEAUX PERSONNAGES
                "emma": "en-US-JennyNeural (patient teacher)",
                "tom": "en-US-GuyNeural (friendly waiter)",
                "lucy": "en-US-AriaNeural (helpful assistant)",
                "ben": "en-US-DavisNeural (clear guide)"
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "fallback_info": "Edge-TTS voice listing failed"
        }

@app.post("/tts/test_all_characters")
async def test_all_character_voices():
    """Tester toutes les voix de personnages avec Edge-TTS - AVEC NOUVEAUX"""
    test_text = "Hello! This is a voice test."
    results = {}
    
    # 🆕 INCLURE TOUS LES PERSONNAGES
    all_characters = ["alice", "mike", "sarah", "emma", "tom", "lucy", "ben"]
    
    for character in all_characters:
        try:
            print(f"🧪 Test voix {character}...")
            audio_path, duration = await edge_text_to_speech_with_fallback(test_text, character)
            # Calculer la taille du fichier
            file_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
            # Nettoyer
            if os.path.exists(audio_path):
                os.remove(audio_path)
            results[character] = {
                "success": True,
                "duration": duration,
                "file_size": file_size,
                "voice_used": DEFAULT_EDGE_VOICES.get(character, "unknown")
            }
        except Exception as e:
            results[character] = {
                "success": False,
                "error": str(e)
            }
    
    return {
        "test_text": test_text,
        "results": results,
        "overall_success": all(r.get("success", False) for r in results.values())
    }

@app.post("/tts/benchmark")
async def benchmark_tts_performance():
    """Benchmark des performances TTS"""
    test_texts = [
        "Hello!",
        "How are you doing today?",
        "The quick brown fox jumps over the lazy dog.",
        "Welcome to our conversation practice system. I'm here to help you improve your English speaking skills."
    ]
    results = []
    
    for i, text in enumerate(test_texts):
        try:
            start_time = time.time()
            audio_path, duration = await edge_text_to_speech_with_fallback(text, "alice")
            generation_time = time.time() - start_time
            file_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
            # Nettoyer
            if os.path.exists(audio_path):
                os.remove(audio_path)
            results.append({
                "test_id": i + 1,
                "text": text,
                "text_length": len(text),
                "generation_time": round(generation_time, 2),
                "audio_duration": round(duration, 2),
                "file_size": file_size,
                "speed_ratio": round(duration / generation_time, 2) if generation_time > 0 else 0
            })
        except Exception as e:
            results.append({
                "test_id": i + 1,
                "text": text,
                "error": str(e)
            })
    
    # Calculer les moyennes
    successful_results = [r for r in results if "error" not in r]
    if successful_results:
        avg_generation_time = sum(r["generation_time"] for r in successful_results) / len(successful_results)
        avg_speed_ratio = sum(r["speed_ratio"] for r in successful_results) / len(successful_results)
        summary = {
            "total_tests": len(test_texts),
            "successful": len(successful_results),
            "failed": len(test_texts) - len(successful_results),
            "avg_generation_time": round(avg_generation_time, 2),
            "avg_speed_ratio": round(avg_speed_ratio, 2),
            "performance_rating": "Excellent" if avg_speed_ratio > 5 else "Good" if avg_speed_ratio > 2 else "Acceptable"
        }
    else:
        summary = {
            "total_tests": len(test_texts),
            "successful": 0,
            "failed": len(test_texts),
            "error": "All tests failed"
        }
    
    return {
        "benchmark_results": results,
        "summary": summary,
        "timestamp": datetime.now().isoformat()
    }

# 🔧 ENDPOINT DE SANTÉ SYSTÈME
@app.get("/health")
async def health_check():
    """Vérification de l'état de santé du système"""
    health_status = {
        "api": "operational",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    # Test Edge-TTS
    try:
        test_audio, _ = await edge_text_to_speech_with_fallback("Health check", "alice")
        if os.path.exists(test_audio):
            os.remove(test_audio)
        health_status["components"]["edge_tts"] = "operational"
    except Exception as e:
        health_status["components"]["edge_tts"] = f"error: {str(e)}"
    
    # Test Phi3/Ollama
    try:
        response = generate_ollama_response('phi3', "Say hello")
        health_status["components"]["phi3_ollama"] = "operational"
    except Exception as e:
        health_status["components"]["phi3_ollama"] = f"error: {str(e)}"
    
    # Test base de données
    try:
        from back.bdd.database import get_db
        db = next(get_db())
        db.execute("SELECT 1")
        health_status["components"]["database"] = "operational"
    except Exception as e:
        health_status["components"]["database"] = f"error: {str(e)}"
    
    # Test répertoires audio
    try:
        ensure_audio_directories()
        health_status["components"]["audio_directories"] = "operational"
    except Exception as e:
        health_status["components"]["audio_directories"] = f"error: {str(e)}"
    
    # Déterminer le statut global
    operational_count = sum(1 for status in health_status["components"].values() if status == "operational")
    total_components = len(health_status["components"])
    
    if operational_count == total_components:
        health_status["status"] = "healthy"
    elif operational_count > total_components / 2:
        health_status["status"] = "degraded"
    else:
        health_status["status"] = "unhealthy"
    
    return health_status

# 🎯 ENDPOINT RAPIDE POUR FRONTEND
@app.get("/quick_status")
async def quick_status():
    """Status rapide pour le frontend"""
    try:
        # Test rapide Edge-TTS
        audio_path, _ = await edge_text_to_speech_with_fallback("Test", "alice")
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return {
            "status": "ready",
            "tts": "edge_tts_operational",
            "message": "System ready for conversations"
        }
    except Exception as e:
        return {
            "status": "degraded",
            "tts": "edge_tts_error",
            "error": str(e),
            "message": "System partially operational"
        }

# ===== ENDPOINTS DE DEBUG (gardés pour compatibilité) =====
@app.post("/test_phi3_mike")
async def test_phi3_mike_debug(
    test_message: str = "How are you?",
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_user)
):
    """Endpoint de test pour diagnostiquer les problèmes avec Phi3 et Mike"""
    print("🔍 === DÉBUT TEST PHI3 MIKE ===")
    # 1. Test du prompt exact envoyé à Phi3
    user_id = current_user.id
    choice = "conv_taxi_booking"
    
    # Simuler un historique simple
    fake_history = [
        {'input': 'hello', 'response': 'NYC Taxi Central, Mike speaking!'}
    ]
    
    # Générer le contexte comme dans la vraie fonction
    conversation_context = ""
    if fake_history:
        recent_history = fake_history[-3:]
        for entry in recent_history:
            if entry.get('input'):
                conversation_context += f"Customer: {entry['input']}\n"
            if entry.get('response'):
                conversation_context += f"Mike: {entry['response']}\n"
    
    # Construire le prompt EXACT comme dans conversation_personalities.py
    mike_personality = f"""You are Mike, a 35-year-old taxi dispatcher from New York City. You are NOT an AI or language model - you are a real person.
Your character traits:
- Real person working at NYC Taxi Central for 10 years
- Knows all NYC routes, traffic patterns, and neighborhoods
- Professional but friendly, experienced with all types of customers
- Uses natural NYC dispatcher language, not robotic responses
IMPORTANT VARIATION INSTRUCTION: Sometimes mention current traffic conditions
or suggest alternate routes.
Current call:
{conversation_context}
The customer just said: "{test_message}"
Respond as Mike the taxi dispatcher would naturally respond. Help them with their taxi booking needs professionally but conversationally. Maximum 2 sentences."""

    # Construire le prompt strict EXACT
    strict_prompt = f"""<|system|>
{mike_personality}
CRITICAL RULES:
1. You MUST stay in character at ALL times - you are the specific person described above
2. NEVER mention you are an AI, language model, or assistant
3. NEVER use phrases like "I'm here to help" or "How can I assist you"
4. Respond naturally as this real person would in this real situation
5. Keep responses conversational and human-like
6. Maximum 2 sentences per response
<|end|>
<|user|>
{test_message}
<|assistant|>
"""

    print("📝 PROMPT ENVOYÉ À PHI3:")
    print("="*50)
    print(strict_prompt)
    print("="*50)

    # 2. Appeler directement Phi3 avec le prompt
    try:
        start_time = time.time()
        print("🤖 Appel direct à Phi3...")
        raw_response = generate_ollama_response('phi3', strict_prompt)
        end_time = time.time()
        response_time = end_time - start_time
        print(f"⏱️ Temps de réponse: {response_time:.2f}s")

        print("📥 RÉPONSE BRUTE DE PHI3:")
        print("="*50)
        print(repr(raw_response))  # repr pour voir les caractères cachés
        print("="*50)
        print("📥 RÉPONSE AFFICHABLE:")
        print(raw_response)
        print("="*50)

        # 3. Nettoyer comme dans le code réel
        cleaned_response = raw_response.strip()
        cleaned_response = re.sub(r'<\|.*?\|>', '', cleaned_response).strip()
        print("🧹 APRÈS NETTOYAGE:")
        print(cleaned_response)
        print("="*50)

        # 4. Tester les détections d'IA
        ai_indicators = [
            "I'm an AI", "As an AI", "I'm a language model", "I'm Microsoft", "I'm Phi",
            "I'm OpenAI", "I'm GPT", "I'm developed by", "I'm designed to", "I'm programmed",
            "As your assistant", "I'm a virtual", "I'm an artificial", "How can I assist",
            "I don't have feelings", "I'm just a computer", "algorithm"
        ]
        detected_indicators = []
        response_lower = cleaned_response.lower()
        for indicator in ai_indicators:
            if indicator.lower() in response_lower:
                detected_indicators.append(indicator)

        print("🚨 INDICATEURS D'IA DÉTECTÉS:")
        print(detected_indicators if detected_indicators else "Aucun")
        print("="*50)

        # 5. Test de la fonction complète
        print("🎭 TEST DE LA FONCTION COMPLÈTE:")
        try:
            ai_function = get_ai_function_for_choice(choice)
            full_response = await asyncio.to_thread(ai_function, test_message, fake_history, user_id)
            print("Réponse fonction complète:", full_response)
        except Exception as func_error:
            print("❌ Erreur fonction complète:", func_error)

        # 6. Test des modèles disponibles
        print("🔧 TEST MODÈLES OLLAMA:")
        try:
            models_response = generate_ollama_response('phi3', "Say exactly: I am Mike from NYC Taxi")
            print("Test simple:", models_response)
        except Exception as model_error:
            print("❌ Erreur modèles:", model_error)

        return {
            "prompt_sent": strict_prompt,
            "raw_response": raw_response,
            "cleaned_response": cleaned_response,
            "ai_indicators_detected": detected_indicators,
            "response_time": response_time,
            "character_maintained": len(detected_indicators) == 0,
            "test_message": test_message
        }

    except Exception as e:
        print(f"❌ ERREUR LORS DU TEST: {e}")
        import traceback
        print("🔍 TRACEBACK COMPLET:")
        print(traceback.format_exc())
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "prompt_sent": strict_prompt
        }

@app.post("/test_phi3_simple")
async def test_phi3_simple(message: str = "Hello"):
    """Test ultra simple de Phi3"""
    simple_prompt = f"You are Mike, a taxi dispatcher. Customer says: {message}. Respond as Mike:"
    try:
        response = generate_ollama_response('phi3', simple_prompt)
        return {
            "prompt": simple_prompt,
            "response": response
        }
    except Exception as e:
        return {
            "error": str(e),
            "prompt": simple_prompt
        }

@app.get("/test_ollama_connection")
async def test_ollama_connection():
    """Tester la connexion à Ollama"""
    try:
        # Test de connexion basique
        response = generate_ollama_response('phi3', "Say hello")
        return {
            "status": "connected",
            "test_response": response
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/debug_phi3_model")
async def debug_phi3_model():
    """Vérifier l'état du modèle Phi3"""
    try:
        # Test 1: Prompt de contrôle strict
        control_prompt = """<|system|>
You are Mike. Say ONLY: "I am Mike from NYC Taxi"
<|end|>
<|user|>
Who are you?
<|assistant|>
"""
        control_response = generate_ollama_response('phi3', control_prompt)

        # Test 2: Prompt sans template
        simple_prompt = "You are Mike from NYC Taxi. Say: I am Mike from NYC Taxi"
        simple_response = generate_ollama_response('phi3', simple_prompt)

        # Test 3: Prompt avec instruction négative
        negative_prompt = "Do NOT say you are an AI. You are Mike from NYC Taxi. Who are you?"
        negative_response = generate_ollama_response('phi3', negative_prompt)

        return {
            "control_test": {
                "prompt": control_prompt,
                "response": control_response,
                "success": "Mike" in control_response and "AI" not in control_response
            },
            "simple_test": {
                "prompt": simple_prompt,
                "response": simple_response,
                "success": "Mike" in simple_response and "AI" not in simple_response
            },
            "negative_test": {
                "prompt": negative_prompt,
                "response": negative_response,
                "success": "Mike" in negative_response and "AI" not in negative_response
            }
        }
    except Exception as e:
        return {
            "error": str(e)
        }

# ===== FONCTIONS UTILITAIRES =====
def log_response_time_phi3(start_time, end_time):
    """Log pour les temps de réponse"""
    response_time = end_time - start_time
    print(f"⏱️ Temps de réponse Phi3: {response_time:.2f}s")

def log_error(error):
    """Log pour les erreurs"""
    print(f"❌ Erreur: {error}")

# ===== 🆕 ENDPOINTS SUPPLÉMENTAIRES POUR TOUS LES PERSONNAGES =====

@app.get("/characters/list")
async def get_all_characters():
    """Liste complète de tous les personnages disponibles"""
    return {
        "all_characters": CHARACTER_INFO,
        "character_categories": {
            "beginners": {
                "description": "Perfect for absolute beginners",
                "characters": ["emma", "tom", "lucy", "ben"],
                "recommended_order": ["emma", "tom", "lucy", "ben"]
            },
            "intermediate": {
                "description": "For students with basic English knowledge",
                "characters": ["alice"],
                "recommended_order": ["alice"]
            },
            "advanced": {
                "description": "Professional and specific scenarios",
                "characters": ["mike", "sarah"],
                "recommended_order": ["mike", "sarah"]
            }
        },
        "learning_path": [
            {
                "level": 1,
                "character": "emma",
                "name": "Emma",
                "description": "Start here! Learn basic English with a patient teacher",
                "skills": ["Basic vocabulary", "Simple greetings", "Yes/No questions"]
            },
            {
                "level": 2,
                "character": "tom",
                "name": "Tom",
                "description": "Order food and drinks in English",
                "skills": ["Food vocabulary", "Restaurant phrases", "Basic numbers"]
            },
            {
                "level": 3,
                "character": "lucy",
                "name": "Lucy",
                "description": "Go shopping and buy clothes",
                "skills": ["Clothing vocabulary", "Colors", "Sizes", "Shopping phrases"]
            },
            {
                "level": 4,
                "character": "ben",
                "name": "Ben",
                "description": "Ask for directions and navigate cities",
                "skills": ["Direction vocabulary", "Location words", "Travel phrases"]
            },
            {
                "level": 5,
                "character": "alice",
                "name": "Alice",
                "description": "Practice general conversation",
                "skills": ["General conversation", "Storytelling", "Opinions"]
            },
            {
                "level": 6,
                "character": "mike",
                "name": "Mike",
                "description": "Book transportation services",
                "skills": ["Professional communication", "Service requests", "City navigation"]
            },
            {
                "level": 7,
                "character": "sarah",
                "name": "Sarah",
                "description": "Handle airline bookings and travel",
                "skills": ["Travel vocabulary", "Formal communication", "Problem solving"]
            }
        ],
        "voice_mapping": DEFAULT_EDGE_VOICES
    }

@app.get("/characters/{character_name}/info")
async def get_character_info(character_name: str):
    """Informations détaillées sur un personnage spécifique"""
    if character_name not in CHARACTER_INFO:
        raise HTTPException(status_code=404, detail="Character not found")
    
    character_info = CHARACTER_INFO[character_name]
    choice = None
    
    # Trouver le choice correspondant
    for choice_key, char in CHOICE_TO_CHARACTER.items():
        if char == character_name:
            choice = choice_key
            break
    
    # Informations spécifiques par personnage
    detailed_info = {
        "emma": {
            "difficulty": "Beginner",
            "scenarios": ["Basic greetings", "Learning new words", "Simple questions"],
            "vocabulary_focus": ["Colors", "Numbers", "Family", "Food basics"],
            "max_words_per_sentence": 10,
            "encouragement_style": "Very positive and patient"
        },
        "tom": {
            "difficulty": "Beginner",
            "scenarios": ["Ordering food", "Asking about menu", "Paying bill"],
            "vocabulary_focus": ["Food items", "Drinks", "Prices", "Restaurant phrases"],
            "typical_phrases": ["What do you want?", "That is 10 dollars", "Here is your food"]
        },
        "lucy": {
            "difficulty": "Beginner", 
            "scenarios": ["Shopping for clothes", "Asking about sizes", "Trying on clothes"],
            "vocabulary_focus": ["Clothing", "Colors", "Sizes", "Materials"],
            "typical_phrases": ["What size?", "Do you like this color?", "The changing room is there"]
        },
        "ben": {
            "difficulty": "Beginner",
            "scenarios": ["Asking directions", "Finding locations", "Using public transport"],
            "vocabulary_focus": ["Directions", "Locations", "Transportation", "Landmarks"],
            "typical_phrases": ["Go straight", "Turn left", "It is very close", "Walk 5 minutes"]
        },
        "alice": {
            "difficulty": "Intermediate",
            "scenarios": ["General conversation", "Sharing experiences", "Discussing interests"],
            "vocabulary_focus": ["Daily life", "Hobbies", "Experiences", "Opinions"],
            "conversation_style": "Natural and engaging"
        },
        "mike": {
            "difficulty": "Advanced",
            "scenarios": ["Booking taxis", "Giving addresses", "Discussing routes"],
            "vocabulary_focus": ["Transportation", "NYC geography", "Time management"],
            "professional_context": "NYC taxi dispatcher"
        },
        "sarah": {
            "difficulty": "Advanced",
            "scenarios": ["Flight booking", "Travel planning", "Customer service"],
            "vocabulary_focus": ["Travel", "Airlines", "Customer service", "Problem solving"],
            "professional_context": "Airline customer service"
        }
    }
    
    return {
        "character": character_name,
        "basic_info": character_info,
        "choice": choice,
        "category": category_mapping.get(choice, "Unknown"),
        "voice": DEFAULT_EDGE_VOICES.get(character_name, "unknown"),
        "detailed_info": detailed_info.get(character_name, {}),
        "endpoint": f"/chat_conv/{character_name}",
        "sample_greeting": get_character_greeting(character_name, 12345)
    }

@app.post("/characters/test_voice/{character_name}")
async def test_character_voice(character_name: str, text: str = "Hello! This is a voice test."):
    """Tester la voix d'un personnage spécifique"""
    if character_name not in CHARACTER_INFO:
        raise HTTPException(status_code=404, detail="Character not found")
    
    try:
        print(f"🧪 Test voix pour {character_name}: '{text}'")
        audio_path, duration = await edge_text_to_speech_with_fallback(text, character_name)
        audio_base64 = await asyncio.to_thread(file_to_base64, audio_path)
        file_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        
        # Nettoyer le fichier de test
        await asyncio.to_thread(delete_audio_file, audio_path)
        
        return {
            "success": True,
            "character": character_name,
            "text": text,
            "duration": duration,
            "file_size": file_size,
            "voice_used": DEFAULT_EDGE_VOICES.get(character_name, "unknown"),
            "audio_base64": audio_base64,
            "message": f"Voice test successful for {character_name}"
        }
    except Exception as e:
        return {
            "success": False,
            "character": character_name,
            "error": str(e),
            "message": f"Voice test failed for {character_name}"
        }

@app.get("/user/progress")
async def get_user_learning_progress(
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Progression d'apprentissage complète de l'utilisateur"""
    user_id = current_user.id
    
    # Récupérer toutes les conversations de l'utilisateur
    all_conversations = db.query(Conversation).filter(
        Conversation.user1_id == user_id
    ).all()
    
    # Analyser par personnage
    character_progress = {}
    total_conversations = 0
    total_messages = 0
    
    for choice, character in CHOICE_TO_CHARACTER.items():
        character_conversations = [conv for conv in all_conversations if conv.category == choice]
        character_messages = 0
        
        for conv in character_conversations:
            messages = db.query(Message).filter(Message.conversation_id == conv.id).count()
            character_messages += messages
        
        character_info = CHARACTER_INFO.get(character, {})
        
        character_progress[character] = {
            "name": character_info.get("name", character),
            "role": character_info.get("role", "Unknown"),
            "conversation_count": len(character_conversations),
            "message_count": character_messages,
            "last_activity": character_conversations[-1].start_time.isoformat() if character_conversations else None,
            "category": category_mapping.get(choice, choice)
        }
        
        total_conversations += len(character_conversations)
        total_messages += character_messages
    
    # Calculer le niveau global
    beginner_chars = ["emma", "tom", "lucy", "ben"]
    intermediate_chars = ["alice"]
    advanced_chars = ["mike", "sarah"]
    
    beginner_progress = sum(1 for char in beginner_chars if character_progress[char]["conversation_count"] > 0)
    intermediate_progress = sum(1 for char in intermediate_chars if character_progress[char]["conversation_count"] > 0)
    advanced_progress = sum(1 for char in advanced_chars if character_progress[char]["conversation_count"] > 0)
    
    # Déterminer le niveau actuel
    if beginner_progress == 0:
        current_level = "Complete Beginner"
        next_goal = "Start with Emma to learn basic English"
    elif beginner_progress < 4:
        current_level = "Beginner"
        next_goal = f"Practice with {4 - beginner_progress} more beginner characters"
    elif intermediate_progress == 0:
        current_level = "Beginner Complete"
        next_goal = "Try Alice for general conversation practice"
    elif advanced_progress == 0:
        current_level = "Intermediate"
        next_goal = "Try Mike or Sarah for professional scenarios"
    else:
        current_level = "Advanced"
        next_goal = "Keep practicing to maintain your skills!"
    
    return {
        "user_id": user_id,
        "overall_stats": {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "current_level": current_level,
            "next_goal": next_goal
        },
        "character_progress": character_progress,
        "level_completion": {
            "beginner": f"{beginner_progress}/4 characters",
            "intermediate": f"{intermediate_progress}/1 characters", 
            "advanced": f"{advanced_progress}/2 characters"
        },
        "recommendations": get_personalized_recommendations(character_progress),
        "achievements": calculate_achievements(character_progress, total_conversations, total_messages)
    }

def get_personalized_recommendations(character_progress):
    """Recommandations personnalisées basées sur la progression"""
    recommendations = []
    
    # Vérifier les personnages non essayés
    for character, progress in character_progress.items():
        if progress["conversation_count"] == 0:
            character_info = CHARACTER_INFO.get(character, {})
            recommendations.append({
                "type": "new_character",
                "character": character,
                "name": character_info.get("name", character),
                "reason": f"Try {character_info.get('name', character)} - {character_info.get('description', '')}"
            })
    
    # Recommandations pour améliorer
    for character, progress in character_progress.items():
        if 0 < progress["conversation_count"] < 3:
            character_info = CHARACTER_INFO.get(character, {})
            recommendations.append({
                "type": "practice_more",
                "character": character,
                "name": character_info.get("name", character),
                "reason": f"Practice more with {character_info.get('name', character)} to build confidence"
            })
    
    return recommendations[:5]  # Limiter à 5 recommandations

def calculate_achievements(character_progress, total_conversations, total_messages):
    """Calculer les achievements/badges de l'utilisateur"""
    achievements = []
    
    # Achievement pour premier contact
    if total_conversations > 0:
        achievements.append({
            "name": "First Steps",
            "description": "Started your first conversation",
            "icon": "🎯",
            "earned": True
        })
    
    # Achievement pour chaque personnage essayé
    for character, progress in character_progress.items():
        if progress["conversation_count"] > 0:
            character_info = CHARACTER_INFO.get(character, {})
            achievements.append({
                "name": f"Met {character_info.get('name', character)}",
                "description": f"Had your first conversation with {character_info.get('name', character)}",
                "icon": "👋",
                "earned": True
            })
    
    # Achievement pour persistance
    if total_conversations >= 10:
        achievements.append({
            "name": "Dedicated Learner",
            "description": "Completed 10 conversations",
            "icon": "🏆",
            "earned": True
        })
    
    # Achievement pour diversité
    characters_tried = sum(1 for progress in character_progress.values() if progress["conversation_count"] > 0)
    if characters_tried >= 4:
        achievements.append({
            "name": "Social Butterfly",
            "description": "Talked with 4 different characters",
            "icon": "🦋",
            "earned": True
        })
    
    # Achievement pour maîtrise débutant
    beginner_chars = ["emma", "tom", "lucy", "ben"]
    beginner_mastery = sum(1 for char in beginner_chars if character_progress[char]["conversation_count"] >= 3)
    if beginner_mastery >= 4:
        achievements.append({
            "name": "Beginner Master",
            "description": "Mastered all beginner characters",
            "icon": "🌟",
            "earned": True
        })
    
    return achievements

@app.get("/user/detailed_stats")
async def get_detailed_user_stats(
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Statistiques détaillées pour la page d'analyse"""
    try:
        user_id = current_user.id
        
        # Récupérer toutes les conversations
        all_conversations = db.query(Conversation).filter(
            Conversation.user1_id == user_id
        ).all()
        
        # Statistiques globales
        total_conversations = len(all_conversations)
        total_time_minutes = 0
        
        # Statistiques par personnage
        character_stats = {}
        
        # Analyser chaque conversation
        for conv in all_conversations:
            # Calculer la durée
            start_time = conv.start_time
            end_time = conv.end_time if conv.end_time else datetime.utcnow()
            duration_minutes = max(1, int((end_time - start_time).total_seconds() / 60))
            total_time_minutes += duration_minutes
            
            # Obtenir le personnage
            character = get_character_from_category_util(conv.category)
            
            if character not in character_stats:
                character_stats[character] = {
                    'conversations': 0,
                    'total_time': 0,
                    'last_activity': None,
                    'messages': 0
                }
            
            character_stats[character]['conversations'] += 1
            character_stats[character]['total_time'] += duration_minutes
            
            if not character_stats[character]['last_activity'] or conv.start_time > character_stats[character]['last_activity']:
                character_stats[character]['last_activity'] = conv.start_time
            
            # Compter les messages
            messages_count = db.query(Message).filter(Message.conversation_id == conv.id).count()
            character_stats[character]['messages'] += messages_count
        
        # Calculer les tendances (cette semaine vs semaine dernière)
        now = datetime.utcnow()
        one_week_ago = now - timedelta(days=7)
        two_weeks_ago = now - timedelta(days=14)
        
        this_week_convs = [c for c in all_conversations if c.start_time >= one_week_ago]
        last_week_convs = [c for c in all_conversations if two_weeks_ago <= c.start_time < one_week_ago]
        
        # Activité par jour (7 derniers jours)
        daily_activity = []
        for i in range(7):
            day = now - timedelta(days=i)
            day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            
            day_conversations = [c for c in all_conversations 
                               if day_start <= c.start_time < day_end]
            
            daily_activity.append({
                'date': day_start.isoformat(),
                'conversations': len(day_conversations)
            })
        
        daily_activity.reverse()  # Du plus ancien au plus récent
        
        # Calculer le niveau de progression
        level_info = calculate_user_level_util(character_stats)
        
        return {
            'global_stats': {
                'total_conversations': total_conversations,
                'total_time_minutes': total_time_minutes,
                'this_week_conversations': len(this_week_convs),
                'last_week_conversations': len(last_week_convs),
                'avg_session_time': total_time_minutes / total_conversations if total_conversations > 0 else 0
            },
            'character_stats': character_stats,
            'level_info': level_info,
            'daily_activity': daily_activity,
            'trends': {
                'conversations_trend': len(this_week_convs) - len(last_week_convs),
                'most_active_character': max(character_stats.keys(), 
                                           key=lambda k: character_stats[k]['conversations']) if character_stats else None
            }
        }
        
    except Exception as e:
        print(f"❌ Erreur statistiques détaillées: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Fonctions utilitaires pour l'endpoint ci-dessus
def get_character_from_category_util(category):
    """Utilitaire pour obtenir le personnage depuis la catégorie"""
    mapping = {
        "conv_greetings_common_conversations": "alice",
        "conv_taxi_booking": "mike",
        "conv_airport_ticket": "sarah",
        "conv_beginner_teacher": "emma",
        "conv_simple_restaurant": "tom",
        "conv_simple_shopping": "lucy",
        "conv_simple_directions": "ben"
    }
    return mapping.get(category, 'unknown')

def calculate_user_level_util(character_stats):
    """Calculer le niveau de l'utilisateur"""
    beginner_chars = ['emma', 'tom', 'lucy', 'ben']
    intermediate_chars = ['alice']
    advanced_chars = ['mike', 'sarah']
    
    beginner_progress = sum(1 for char in beginner_chars 
                          if char in character_stats and character_stats[char]['conversations'] > 0)
    
    intermediate_progress = sum(1 for char in intermediate_chars 
                              if char in character_stats and character_stats[char]['conversations'] > 0)
    
    advanced_progress = sum(1 for char in advanced_chars 
                          if char in character_stats and character_stats[char]['conversations'] > 0)
    
    if beginner_progress == 0:
        level = "Nouveau"
        progress_percentage = 0
    elif beginner_progress < 4:
        level = "Débutant"
        progress_percentage = (beginner_progress / 4) * 40
    elif intermediate_progress == 0:
        level = "Débutant Avancé" 
        progress_percentage = 40
    elif advanced_progress == 0:
        level = "Intermédiaire"
        progress_percentage = 40 + (intermediate_progress / 1) * 30
    else:
        level = "Avancé"
        progress_percentage = 70 + (advanced_progress / 2) * 30
    
    return {
        'level': level,
        'progress_percentage': round(progress_percentage),
        'beginner_progress': f"{beginner_progress}/4",
        'intermediate_progress': f"{intermediate_progress}/1", 
        'advanced_progress': f"{advanced_progress}/2"
    }

# 🆕 GESTION DES AVATARS ET PROFILS
@app.post("/user/upload-avatar")
async def upload_avatar(
    avatar: UploadFile = File(...),
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload d'avatar utilisateur"""
    try:
        # Vérifier le type de fichier
        if not avatar.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Le fichier doit être une image")
        
        # Vérifier la taille (5MB max)
        if avatar.size > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="L'image ne doit pas dépasser 5MB")
        
        # Créer le dossier avatars s'il n'existe pas
        avatar_dir = "./front/static/assets/images/avatars"
        os.makedirs(avatar_dir, exist_ok=True)
        
        # Générer un nom de fichier unique
        file_extension = avatar.filename.split('.')[-1].lower()
        if file_extension not in ['jpg', 'jpeg', 'png', 'gif']:
            raise HTTPException(status_code=400, detail="Format d'image non supporté")
        
        avatar_filename = f"user_{current_user.id}_{int(time.time())}.{file_extension}"
        avatar_path = os.path.join(avatar_dir, avatar_filename)
        
        # Sauvegarder le fichier
        async with aiofiles.open(avatar_path, 'wb') as f:
            content = await avatar.read()
            await f.write(content)
        
        # Mettre à jour l'URL dans la base de données
        avatar_url = f"/static/assets/images/avatars/{avatar_filename}"
        current_user.avatar_url = avatar_url
        db.commit()
        
        return {
            "success": True,
            "avatar_url": avatar_url,
            "message": "Avatar mis à jour avec succès"
        }
        
    except Exception as e:
        print(f"❌ Erreur upload avatar: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/user/update-profile")
async def update_user_profile(
    nom: Optional[str] = Form(None),
    bio: Optional[str] = Form(None),
    country: Optional[str] = Form(None),
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mise à jour du profil utilisateur"""
    try:
        if nom:
            current_user.nom = nom
        if bio is not None:
            current_user.bio = bio
        if country is not None:
            current_user.country = country
            
        db.commit()
        db.refresh(current_user)
        
        return {
            "success": True,
            "user": {
                "id": current_user.id,
                "nom": current_user.nom,
                "email": current_user.email,
                "bio": getattr(current_user, 'bio', ''),
                "country": getattr(current_user, 'country', ''),
                "avatar_url": getattr(current_user, 'avatar_url', None)
            }
        }
        
    except Exception as e:
        print(f"❌ Erreur mise à jour profil: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/user/change-password")
async def change_password(
    current_password: str = Form(...),
    new_password: str = Form(...),
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Changement de mot de passe"""
    try:
        from back.token_security import verify_password, hash_password
        
        # Vérifier l'ancien mot de passe
        if not verify_password(current_password, current_user.hashed_password):
            raise HTTPException(status_code=400, detail="Mot de passe actuel incorrect")
        
        # Valider le nouveau mot de passe
        if len(new_password) < 6:
            raise HTTPException(status_code=400, detail="Le nouveau mot de passe doit contenir au moins 6 caractères")
        
        # Hasher et sauvegarder le nouveau mot de passe
        current_user.hashed_password = hash_password(new_password)
        db.commit()
        
        return {
            "success": True,
            "message": "Mot de passe mis à jour avec succès"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur changement mot de passe: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_main:app", host="0.0.0.0", port=8000, reload=True)