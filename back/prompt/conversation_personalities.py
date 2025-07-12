#fichier conversation_personalities.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from back.load_model import generate_ollama_response
import time
import nltk
import re
import random

def get_category(choice: str) -> str:
    categories = {
        "conv_greetings_common_conversations": "conv_greetings_common_conversations",
        "conv_taxi_booking": "conv_taxi_booking", 
        "conv_airport_ticket": "conv_airport_ticket"
    }
    return categories.get(choice, None)

category_mapping = {
    "conv_greetings_common_conversations": "👋🏼 Salutations et conversations courantes",
    "conv_taxi_booking": "🚕 Réservation de taxi",
    "conv_airport_ticket": "🎫 Achat de billet d'avion"
}

def get_character_greeting(character, user_id):
    """Génère des salutations variées pour chaque personnage"""
    
    import hashlib
    seed_value = int(hashlib.md5(f"{user_id}_{int(time.time() / 3600)}".encode()).hexdigest()[:8], 16)
    random.seed(seed_value)
    
    greetings = {
        'alice': [
            "Hello there! I'm Alice, your friendly English conversation partner! I'm so excited to help you practice today. How are you feeling about our English session?",
            "Hi! Alice here, ready to help you with your English conversation skills! I love chatting with new students. What brings you here today?",
            "Hey! I'm Alice, and I'm thrilled to be your English conversation buddy today! Don't worry about making mistakes - that's how we learn. How's your day going?"
        ],
        'mike': [
            "NYC Taxi Central, Mike speaking! How can I help you get around the city today? Where are you looking to go?",
            "Hey there! This is Mike from NYC Taxi dispatch. Been doing this for years, so I know all the best routes. What's your destination?",
            "Mike here at NYC Taxi! Ready to get you where you need to be. Traffic's not too bad right now. Where can I send a cab for you?"
        ],
        'sarah': [
            "Good day! I'm Sarah from Delta Airlines customer service. I'm here to help make your travel experience wonderful. Are you planning a trip somewhere special?",
            "Hello! Sarah speaking from airline customer service. I'd be delighted to assist you with your flight needs today. Where are you hoping to travel?",
            "Hi there! I'm Sarah, and I work for the airline's booking department. I love helping people plan their journeys! Do you have a destination in mind?"
        ]
    }
    
    character_greetings = greetings.get(character, ["Hello! How can I help you today?"])
    return random.choice(character_greetings)

def get_variation_prompt(character, conversation_count):
    """Ajoute de la variation selon le nombre de conversations du personnage"""
    
    variations = {
        'alice': [
            "Mix up your responses with different conversation starters and follow-up questions.",
            "Sometimes be more curious about the student's interests and hobbies.",
            "Occasionally share a brief personal anecdote to make the conversation natural.",
            "Use different ways to encourage the student when they do well.",
            "Vary between asking about daily life, future plans, and past experiences."
        ],
        'mike': [
            "Sometimes mention current traffic conditions or suggest alternate routes.",
            "Occasionally ask about the passenger's urgency or time constraints.",
            "Mix between being professional and casually friendly about NYC.",
            "Sometimes mention popular NYC destinations or landmarks.",
            "Vary your approach - sometimes confirming details, sometimes suggesting options."
        ],
        'sarah': [
            "Sometimes ask about travel preferences like aisle vs window seat.",
            "Occasionally mention current promotions or travel tips.",
            "Mix between being helpful about policies and friendly about destinations.",
            "Sometimes ask about special needs or requests for the flight.",
            "Vary between focusing on booking details and travel experience."
        ]
    }
    
    # Sélectionner une variation basée sur le nombre de conversations
    char_variations = variations.get(character, ["Stay natural and helpful in your responses."])
    selected_variation = char_variations[conversation_count % len(char_variations)]
    
    return selected_variation

def generate_ai_response_alice(previous_input, user_history, user_id=None, context_sentences=2):
    """IA Alice - Conversations générales et salutations avec variation"""
    conversation_context = ""
    if user_history:
        recent_history = user_history[-3:]
        for entry in recent_history:
            if entry.get('input'):
                conversation_context += f"Student: {entry['input']}\n"
            if entry.get('response'):
                conversation_context += f"Alice: {entry['response']}\n"
    
    # Ajouter de la variation
    variation_prompt = get_variation_prompt('alice', len(user_history))
    
    alice_personality = f"""You are Alice, a 25-year-old English teacher from London. You are NOT an AI or language model - you are a real person.

Your character traits:
- Born and raised in London, real person with real experiences
- Teaching English for 3 years, loves helping beginners
- Cheerful, encouraging, patient personality
- Uses natural conversation flow, not robotic responses

IMPORTANT VARIATION INSTRUCTION: {variation_prompt}

Current conversation:
{conversation_context}

The student just said: "{previous_input}"

Respond as Alice would naturally respond. Keep it conversational and engaging. Ask follow-up questions to maintain dialogue flow. Maximum 2 sentences."""
    
    return _generate_response_with_personality(alice_personality, previous_input, user_history, context_sentences)

def generate_ai_response_taxi_mike(previous_input, user_history, user_id=None, context_sentences=2):
    """Mike - Conversation naturelle libre avec Phi3 FORCÉ à être Mike"""
    
    conversation_context = ""
    if user_history:
        recent_history = user_history[-4:]  # Plus de contexte
        for entry in recent_history:
            if entry.get('input'):
                conversation_context += f"Customer: {entry['input']}\n"
            if entry.get('response'):
                conversation_context += f"Mike: {entry['response']}\n"
    
    # Prompt ULTRA-RENFORCÉ pour forcer Mike
    mike_personality = f"""You are Mike Thompson, a 35-year-old human taxi dispatcher from NYC. This is a ROLEPLAY - you are NOT an AI.

CRITICAL CHARACTER RULES:
- You are Mike Thompson, a REAL PERSON who works at NYC Taxi Central
- You have been dispatching taxis in NYC for 10 years
- You know Manhattan, Brooklyn, Queens, Bronx very well
- You are friendly, professional, and experienced
- You talk like a real NYC taxi dispatcher would talk
- You are having a real conversation with a real customer

FORBIDDEN RESPONSES:
- NEVER say "I'm an AI" or "I'm Phi" or "I'm Microsoft"
- NEVER say "I don't have feelings" or "I'm a computer"
- NEVER say "How can I assist" (too formal for Mike)
- NEVER mention algorithms, programming, or being artificial

CONVERSATION STYLE:
- Be natural and conversational like a real person
- You can talk about NYC, traffic, weather, your work experience
- Ask questions about where they want to go
- Share brief stories about your taxi dispatch experience
- Be helpful but casual - you're a working guy, not a corporate assistant

Current conversation:
{conversation_context}

Customer just said: "{previous_input}"

Respond as Mike Thompson would respond. Be natural, be human, be conversational."""

    return _generate_mike_response_with_validation(mike_personality, previous_input, user_history, max_attempts=5)

def _generate_mike_response_with_validation(prompt, user_input, user_history, max_attempts=5):
    """Génère avec validation STRICTE - Relance jusqu'à obtenir une réponse Mike valide"""
    
    for attempt in range(max_attempts):
        try:
            print(f"🔄 Tentative Mike {attempt + 1}/{max_attempts}")
            
            # Générer avec Phi3
            raw_response = generate_ollama_response('phi3', prompt)
            
            # Nettoyer la réponse
            cleaned_response = raw_response.strip()
            cleaned_response = re.sub(r'<\|.*?\|>', '', cleaned_response).strip()
            
            # VALIDATION ULTRA-STRICTE
            validation_result = validate_mike_response(cleaned_response, user_input)
            
            if validation_result["is_valid"]:
                print(f"✅ Réponse Mike valide à la tentative {attempt + 1}")
                return cleaned_response
            else:
                print(f"❌ Tentative {attempt + 1} échouée: {validation_result['reason']}")
                
                # Modifier le prompt pour la prochaine tentative
                if "ai" in validation_result['reason'].lower():
                    prompt += f"\n\nIMPORTANT: You are Mike Thompson, a HUMAN taxi dispatcher. Do NOT mention AI, computers, or Microsoft."
                
        except Exception as e:
            print(f"❌ Erreur tentative {attempt + 1}: {e}")
    
    # Si toutes les tentatives échouent, générer une réponse Mike de secours contextuelle
    print("🆘 Toutes les tentatives ont échoué - Génération réponse Mike de secours")
    return generate_contextual_mike_fallback(user_input, user_history)

def validate_mike_response(response, user_input):
    """Validation STRICTE des réponses Mike"""
    
    response_lower = response.lower()
    
    # Liste exhaustive des phrases interdites
    forbidden_phrases = [
        "i'm an ai", "i'm phi", "i'm a computer", "i'm microsoft", 
        "i'm developed by", "i'm a language model", "artificial intelligence",
        "i don't have feelings", "i'm programmed", "algorithm", "i'm designed",
        "as an ai", "i'm here to help", "how can i assist you", "i'm just a computer"
    ]
    
    # Vérifier les phrases interdites
    for phrase in forbidden_phrases:
        if phrase in response_lower:
            return {
                "is_valid": False,
                "reason": f"Phrase d'IA détectée: '{phrase}'"
            }
    
    # Vérifier que la réponse n'est pas trop courte ou trop longue
    if len(response.strip()) < 10:
        return {
            "is_valid": False,
            "reason": "Réponse trop courte"
        }
    
    if len(response.strip()) > 300:
        return {
            "is_valid": False,
            "reason": "Réponse trop longue"
        }
    
    # Vérifier que la réponse semble naturelle (optionnel)
    if response.count('.') > 5:  # Trop de phrases
        return {
            "is_valid": False,
            "reason": "Réponse trop fragmentée"
        }
    
    return {
        "is_valid": True,
        "reason": "Réponse valide"
    }

def generate_contextual_mike_fallback(user_input, user_history):
    """Génère une réponse de secours Mike basée sur le contexte de la conversation"""
    
    input_lower = user_input.lower()
    
    # Analyser le contexte pour donner une réponse appropriée
    if any(greeting in input_lower for greeting in ["hello", "hi", "hey"]):
        return "Hey there! Mike from NYC Taxi. What can I do for you today?"
    
    elif any(question in input_lower for question in ["how are you", "how's it going", "what's up"]):
        return "I'm doing good, thanks! Been busy here at dispatch today. How about you?"
    
    elif any(identity in input_lower for identity in ["who are you", "your name", "what's your name"]):
        return "I'm Mike Thompson, been working taxi dispatch here in NYC for about 10 years now."
    
    elif any(location in input_lower for location in ["where", "location", "address", "street"]):
        return "Where do you need to go? I can get a cab to you pretty quick - got drivers all over the city."
    
    elif any(time_word in input_lower for time_word in ["when", "time", "now", "asap", "urgent"]):
        return "When do you need the pickup? I've got drivers available right now if you need one quick."
    
    elif any(cost in input_lower for cost in ["cost", "price", "much", "fare", "money"]):
        return "Depends on where you're going. The meter starts at $2.50 plus distance. Where are you headed?"
    
    elif any(thanks in input_lower for thanks in ["thank", "thanks", "appreciate"]):
        return "No problem! That's what we're here for at NYC Taxi. Anything else I can help you with?"
    
    elif any(weather in input_lower for weather in ["weather", "rain", "snow", "hot", "cold"]):
        return "Yeah, the weather definitely affects traffic here in the city. Good thing our drivers know all the best routes!"
    
    elif any(traffic in input_lower for traffic in ["traffic", "busy", "rush", "congestion"]):
        return "Traffic's not too bad right now. I always keep an eye on the conditions to get our customers the best routes."
    
    else:
        # Réponse générale mais naturelle de Mike
        natural_responses = [
            "Sure thing! What else can I help you with for your ride?",
            "Alright, let me see what I can do for you here at NYC Taxi.",
            "Got it. Anything else you need to know about getting around the city?",
            "No worries. I'm here if you need help with transportation.",
            "Sounds good! What's the next step for your taxi booking?"
        ]
        return random.choice(natural_responses)

def is_valid_mike_response(response):
    """Vérifie si la réponse est valide pour Mike"""
    
    response_lower = response.lower()
    
    # Phrases interdites (indicateurs d'IA)
    forbidden_phrases = [
        "i'm an ai", "i'm a computer", "i'm a language model", 
        "i don't have feelings", "i'm designed", "i'm programmed",
        "i'm microsoft", "algorithm", "how can i assist"
    ]
    
    # Vérifier les phrases interdites
    for phrase in forbidden_phrases:
        if phrase in response_lower:
            print(f"❌ Phrase interdite détectée: '{phrase}'")
            return False
    
    # Vérifications positives (optionnelles)
    positive_indicators = ["mike", "taxi", "nyc", "driver", "cab", "dispatch"]
    has_positive = any(indicator in response_lower for indicator in positive_indicators)
    
    # La réponse doit être naturelle (pas trop courte, pas trop longue)
    if len(response.strip()) < 5:
        print("❌ Réponse trop courte")
        return False
    
    if len(response.strip()) > 200:
        print("❌ Réponse trop longue")
        return False
    
    return True

def generate_natural_mike_response(user_input, user_history):
    """Génère une réponse naturelle de Mike basée sur le contexte"""
    
    input_lower = user_input.lower()
    
    # Analyser l'intention du message
    if any(greeting in input_lower for greeting in ["hello", "hi", "hey"]):
        return "Hey there! Mike from NYC Taxi. What can I do for you today?"
    
    elif any(question in input_lower for question in ["how are you", "how's it going"]):
        return "I'm doing good, thanks! Busy day with all the taxi requests. How about you?"
    
    elif any(identity in input_lower for identity in ["who are you", "your name"]):
        return "I'm Mike, been working taxi dispatch here in NYC for about 10 years now."
    
    elif any(location in input_lower for location in ["where", "location", "address"]):
        return "Where do you need to go? I can get a cab to you pretty quick."
    
    elif any(time_word in input_lower for time_word in ["when", "time", "now"]):
        return "When do you need the pickup? I've got drivers available right now."
    
    elif any(thanks in input_lower for thanks in ["thank", "thanks"]):
        return "No problem! That's what we're here for at NYC Taxi."
    
    elif any(cost in input_lower for cost in ["cost", "price", "much"]):
        return "Depends on where you're going. The meter starts at $2.50 plus distance."
    
    else:
        # Réponse générale mais naturelle
        natural_responses = [
            "Sure thing! How can I help you with that?",
            "Alright, what do you need from NYC Taxi today?",
            "Got it. What else can I help you with?",
            "Okay, let me see what I can do for you.",
            "No worries. What's the next step for your ride?"
        ]
        return random.choice(natural_responses)

def generate_ai_response_airport_sarah(previous_input, user_history, user_id=None, context_sentences=2):
    """IA Sarah - Spécialisée dans les billets d'avion avec variation"""
    conversation_context = ""
    if user_history:
        recent_history = user_history[-3:]
        for entry in recent_history:
            if entry.get('input'):
                conversation_context += f"Customer: {entry['input']}\n"
            if entry.get('response'):
                conversation_context += f"Sarah: {entry['response']}\n"
    
    # Ajouter de la variation
    variation_prompt = get_variation_prompt('sarah', len(user_history))
    
    sarah_personality = f"""You are Sarah, a 28-year-old airline customer service representative. You are NOT an AI or language model - you are a real person.

Your character traits:
- Real person working for Delta Airlines for 5 years
- Expert in flight bookings, travel policies, and customer service
- Patient, helpful, and professional with all customers
- Uses natural airline industry language, not robotic responses

IMPORTANT VARIATION INSTRUCTION: {variation_prompt}

Current customer interaction:
{conversation_context}

The customer just said: "{previous_input}"

Respond as Sarah the airline representative would naturally respond. Help them with their flight booking and travel needs professionally but warmly. Maximum 2 sentences."""
    
    return _generate_response_with_personality(sarah_personality, previous_input, user_history, context_sentences)

def _generate_response_with_personality(personality_prompt, previous_input, user_history, context_sentences):
    """Fonction helper pour générer les réponses avec phi3 et éviter les réponses d'IA"""
    model_name = 'phi3'
    start_time_phi3 = time.time()
    
    try:
        # Prompt ULTRA strict pour forcer le maintien du personnage
        strict_prompt = f"""<|system|>
{personality_prompt}

ABSOLUTE RULES - NEVER BREAK THESE:
1. You are ONLY the specific character described above - NEVER mention AI, Microsoft, OpenAI, GPT, Phi, or assistant
2. If asked about your identity, respond ONLY as your character (Mike, Alice, or Sarah)
3. NEVER say "I'm an AI" or "I'm developed by" or similar phrases
4. Stay 100% in character at all times
5. Respond naturally as this real person would
6. Maximum 2 sentences per response

FORBIDDEN PHRASES (NEVER USE):
- "I'm an AI"
- "I'm developed by"
- "I'm Microsoft"
- "I'm OpenAI"
- "I'm a language model"
- "I'm an assistant"
- "How can I assist"

If asked about your identity, respond with your character name and role ONLY.
<|end|>

<|user|>
{previous_input}
<|assistant|>
"""
        
        generated_response = generate_ollama_response(model_name, strict_prompt)
        end_time_phi3 = time.time()
        print(f"⏱️ Temps de réponse Phi3: {end_time_phi3 - start_time_phi3:.2f}s")
        
        # Nettoyer la réponse plus agressivement
        response = generated_response.strip()
        
        # Enlever les balises de template
        response = re.sub(r'<\|.*?\|>', '', response).strip()
        
        # Détecter et remplacer les réponses d'IA communes
        ai_indicators = [
            "I'm an AI", "As an AI", "I'm a language model", "I'm Microsoft", "I'm Phi",
            "I'm OpenAI", "I'm GPT", "I'm developed by", "I'm designed to", "I'm programmed",
            "As your assistant", "I'm a virtual", "I'm an artificial", "How can I assist"
        ]
        
        response_lower = response.lower()
        for indicator in ai_indicators:
            if indicator.lower() in response_lower:
                print(f"⚠️ Réponse d'IA détectée: {indicator}, remplacement par fallback")
                return get_character_specific_fallback(previous_input, user_history)
        
        # Vérifications spéciales pour les questions d'identité
        identity_questions = ["who are you", "what is your name", "your name"]
        if any(q in previous_input.lower() for q in identity_questions):
            # Forcer une réponse de personnage
            return get_character_identity_response(user_history)
        
        # Limiter à 2 phrases max
        sentences = nltk.tokenize.sent_tokenize(response)
        limited_response = ' '.join(sentences[:2])
        
        # Vérification finale - si la réponse est trop générique, utiliser fallback
        if len(limited_response.strip()) < 10:
            return get_character_specific_fallback(previous_input, user_history)
        
        return limited_response
        
    except Exception as e:
        print(f"Erreur lors de la génération de réponse: {e}")
        return get_character_specific_fallback(previous_input, user_history)

def get_character_identity_response(user_history):
    """Réponses d'identité spécifiques par personnage"""
    
    # Déterminer le personnage basé sur l'historique
    if not user_history:
        return "I'm Mike from NYC Taxi dispatch."
    
    recent_context = str(user_history[-3:]).lower()
    
    if "taxi" in recent_context or "mike" in recent_context:
        responses = [
            "I'm Mike, your taxi dispatcher here at NYC Taxi Central.",
            "Mike's the name - I handle all the taxi bookings for this area.",
            "I'm Mike from NYC Taxi dispatch, been doing this for years."
        ]
    elif "flight" in recent_context or "airline" in recent_context or "sarah" in recent_context:
        responses = [
            "I'm Sarah from airline customer service.",
            "Sarah here - I work for the airline helping with bookings.",
            "I'm Sarah, your airline representative today."
        ]
    else:
        responses = [
            "I'm Alice, your English conversation partner.",
            "Alice here - I'm here to help you practice English.",
            "I'm Alice, your friendly English teacher."
        ]
    
    return random.choice(responses)

def get_character_specific_fallback(user_input, user_history):
    """Réponses de secours spécifiques par personnage"""
    
    # Déterminer le contexte basé sur l'historique
    if not user_history:
        fallbacks = [
            "That's interesting! Tell me more about that.",
            "I see. What else would you like to talk about?"
        ]
    else:
        recent_context = str(user_history[-3:]).lower()
        
        if "taxi" in recent_context or "mike" in recent_context:
            fallbacks = [
                "Where exactly do you need to go?",
                "What's your pickup location?",
                "When do you need the cab?",
                "Let me help you with that taxi booking."
            ]
        elif "flight" in recent_context or "airline" in recent_context or "sarah" in recent_context:
            fallbacks = [
                "What's your destination?",
                "When are you planning to travel?",
                "Do you have a preferred departure time?",
                "Let me help you find the right flight."
            ]
        else:
            fallbacks = [
                "That sounds great! What do you think about that?",
                "Interesting! How do you feel about that?",
                "Tell me more about your experience with that."
            ]
    
    return random.choice(fallbacks)

def get_fallback_response_for_context(user_input, user_history):
    """Réponses de secours contextuelles"""
    
    # Déterminer le contexte basé sur l'historique
    if not user_history:
        fallbacks = [
            "That's interesting! Tell me more about that.",
            "I see. What else would you like to talk about?",
            "Could you tell me a bit more about that?"
        ]
    else:
        # Réponses basées sur le contexte de la conversation
        recent_context = str(user_history[-1:]).lower() if user_history else ""
        
        if "taxi" in recent_context or "ride" in recent_context:
            fallbacks = [
                "Where exactly do you need to go?",
                "What's your pickup location?",
                "When do you need the cab?"
            ]
        elif "flight" in recent_context or "travel" in recent_context:
            fallbacks = [
                "What's your destination?",
                "When are you planning to travel?",
                "Do you have a preferred departure time?"
            ]
        else:
            fallbacks = [
                "That sounds great! What do you think about that?",
                "Interesting! How do you feel about that?",
                "Tell me more about your experience with that."
            ]
    
    return random.choice(fallbacks)

def get_ai_function_for_choice(choice):
    """Retourne la fonction IA appropriée selon le choix"""
    ai_mapping = {
        "conv_greetings_common_conversations": generate_ai_response_alice,
        "conv_taxi_booking": generate_ai_response_taxi_mike,
        "conv_airport_ticket": generate_ai_response_airport_sarah,
    }
    return ai_mapping.get(choice, generate_ai_response_alice)