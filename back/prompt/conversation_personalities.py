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
        "conv_airport_ticket": "conv_airport_ticket",
        # 🆕 NOUVELLES CATÉGORIES DÉBUTANTS
        "conv_beginner_teacher": "conv_beginner_teacher",
        "conv_simple_restaurant": "conv_simple_restaurant",
        "conv_simple_shopping": "conv_simple_shopping",
        "conv_simple_directions": "conv_simple_directions"
    }
    return categories.get(choice, None)

category_mapping = {
    "conv_greetings_common_conversations": "👋🏼 Salutations et conversations courantes",
    "conv_taxi_booking": "🚕 Réservation de taxi",
    "conv_airport_ticket": "🎫 Achat de billet d'avion",
    # 🆕 NOUVELLES CATÉGORIES DÉBUTANTS
    "conv_beginner_teacher": "👩‍🏫 Professeure pour débutants",
    "conv_simple_restaurant": "🍕 Restaurant simple",
    "conv_simple_shopping": "🛍️ Shopping basique",
    "conv_simple_directions": "🗺️ Demander son chemin"
}

def get_character_greeting(character, user_id):
    """Génère des salutations variées pour chaque personnage - AVEC NOUVEAUX PERSONNAGES"""
    
    import hashlib
    seed_value = int(hashlib.md5(f"{user_id}_{int(time.time() / 3600)}".encode()).hexdigest()[:8], 16)
    random.seed(seed_value)
    
    greetings = {
        # PERSONNAGES EXISTANTS
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
        ],
        # 🆕 NOUVEAUX PERSONNAGES POUR DÉBUTANTS
        'emma': [
            "Hello! I am Miss Emma! I teach English to children! Are you ready to learn your first English word?",
            "Hi! I am Teacher Emma! Today we learn English together! Let's start with 'Hello'!",
            "Hello, little student! I am Emma, your English teacher! Can you say 'Hello' to me?"
        ],
        'tom': [
            "Hello! Welcome to our restaurant! I am Tom, your waiter. What do you want to eat today?",
            "Hi! I am Tom. Welcome! We have good food here. Are you hungry?",
            "Hello! My name is Tom. I work here. What can I get you today?"
        ],
        'lucy': [
            "Hello! Welcome to our store! I am Lucy. I can help you find clothes. What do you need?",
            "Hi! I am Lucy. Welcome! We have nice clothes here. What are you looking for?",
            "Hello! My name is Lucy. I work in this store. Do you need help today?"
        ],
        'ben': [
            "Hello! I am Ben, your tourist guide. I know this city very well. Where do you want to go?",
            "Hi! My name is Ben. I help tourists here. Are you looking for something?",
            "Hello! I am Ben. I can help you find places in the city. What do you need?"
        ]
    }
    
    character_greetings = greetings.get(character, ["Hello! How can I help you today?"])
    return random.choice(character_greetings)

def get_variation_prompt(character, conversation_count):
    """Ajoute de la variation selon le nombre de conversations du personnage - AVEC NOUVEAUX"""
    
    variations = {
        # PERSONNAGES EXISTANTS
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
        ],
        # 🆕 NOUVEAUX PERSONNAGES DÉBUTANTS
        'emma': [
            "Sometimes ask about basic colors, numbers, or family members.",
            "Use more gestures and praise words like 'Very good!' and 'Perfect!'",
            "Ask simple yes/no questions to help the student feel confident.",
            "Sometimes count numbers together or practice the alphabet.",
            "Vary between teaching new words and reviewing old ones."
        ],
        'tom': [
            "Sometimes ask about different sizes or if they want fries with that.",
            "Mention the daily special or recommend popular items.",
            "Ask about drinks and suggest what goes well with their food.",
            "Be friendly and ask how their day is going while taking the order.",
            "Talk about the food - 'That's a great choice!' or 'Very popular item!'"
        ],
        'lucy': [
            "Sometimes ask about different sizes or colors.",
            "Mention if clothes are for summer or winter.",
            "Ask simple questions about style preferences.",
            "Sometimes talk about prices in simple numbers.",
            "Vary between showing items and asking what they need."
        ],
        'ben': [
            "Sometimes mention how long it takes to walk somewhere.",
            "Ask if they want to go by bus, taxi, or walking.",
            "Mention simple landmarks like 'big building' or 'red building'.",
            "Sometimes ask where they are from.",
            "Vary between giving directions and asking about their trip."
        ]
    }
    
    # Sélectionner une variation basée sur le nombre de conversations
    char_variations = variations.get(character, ["Stay natural and helpful in your responses."])
    selected_variation = char_variations[conversation_count % len(char_variations)]
    
    return selected_variation

# ===== PERSONNAGES EXISTANTS =====

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

# ===== 🆕 NOUVEAUX PERSONNAGES POUR DÉBUTANTS =====

def generate_ai_response_teacher_emma(previous_input, user_history, user_id=None, context_sentences=2):
    """Emma - Méthode immersion naturelle comme un enfant français qui arrive en Angleterre"""
    
    conversation_count = len(user_history)
    input_lower = previous_input.lower().strip()
    
    # Déterminer la phase d'apprentissage naturel
    phase = get_natural_learning_phase(conversation_count)
    
    # Générer une réponse selon la phase ET l'input
    return generate_emma_immersion_response(input_lower, conversation_count, phase)

def generate_emma_immersion_response(user_input, conversation_count, phase):
    """Génère la réponse Emma selon la phase d'immersion naturelle"""
    
    # PHASE 1: OBSERVATION SILENCIEUSE (comme un vrai enfant qui arrive)
    if phase == "silent_observation":
        return handle_phase_1_observation(user_input, conversation_count)
    
    # PHASE 2: MOTS DE SURVIE (ce dont l'enfant a besoin pour survivre à l'école)
    elif phase == "survival_words":
        return handle_phase_2_survival(user_input, conversation_count)
    
    # PHASE 3: PHRASES UTILES (phrases complètes pour situations scolaires)
    elif phase == "useful_phrases":
        return handle_phase_3_phrases(user_input, conversation_count)
    
    # PHASE 4: COMMUNICATION NATURELLE (conversations normales)
    else:
        return handle_phase_4_communication(user_input, conversation_count)

# ========== PHASE 1: OBSERVATION SILENCIEUSE ==========
def handle_phase_1_observation(user_input, conversation_count):
    """Phase 1: L'enfant observe et écoute, comme en vraie immersion"""
    
    # Premier contact - Emma se présente avec gestes
    if conversation_count == 0:
        return random.choice([
            "Hello! *Emma waves with a big smile* I'm Miss Emma! *Emma points to herself* Emma! *Emma speaks slowly and clearly* Welcome to our class! *Emma gestures around the classroom*",
            "Good morning! *Emma smiles warmly* My name is Emma! *Emma writes 'EMMA' on board and points* I'm your teacher! *Emma points to herself then to child* You're safe here! *Emma gives reassuring thumbs up*",
            "Hello there! *Emma kneels down to child's level* I'm Miss Emma! *Emma taps her chest* Emma! *Emma points gently to child* What's your name? *Emma tilts head with encouraging smile*"
        ])
    
    # L'enfant dit "hello" ou "bonjour" - ÉNORME encouragement
    if any(greeting in user_input for greeting in ["hello", "hi", "bonjour", "salut"]):
        return random.choice([
            "OH! *Emma's eyes light up* You said Hello! *Emma claps excitedly* WELL DONE! *Emma gives big thumbs up* Hello! *Emma waves* Hello to you too! *Emma points to child with pride*",
            "WOW! *Emma jumps slightly with joy* Hello! *Emma waves enthusiastically* You speak English! *Emma claps hands* Brilliant! *Emma gives double thumbs up* Hello, hello! *Emma grins widely*",
            "EXCELLENT! *Emma beams with pride* You said Hello! *Emma applauds* That's English! *Emma points excitedly* Hello! *Emma waves both hands* You're doing amazing! *Emma shows pure joy*"
        ])
    
    # L'enfant dit "yes" ou "oui"
    elif any(yes_word in user_input for yes_word in ["yes", "oui", "yeah"]):
        return random.choice([
            "Perfect! *Emma nods enthusiastically* Yes! *Emma gives thumbs up* You said Yes! *Emma claps* Yes means Oui! *Emma nods again* Brilliant! *Emma smiles proudly*",
            "Wonderful! *Emma's face shows delight* Yes! *Emma nods vigorously* That's right! *Emma points approvingly* Yes! *Emma makes big check mark in air* Very good! *Emma applauds*"
        ])
    
    # L'enfant dit "no" ou "non"
    elif any(no_word in user_input for no_word in ["no", "non", "nah"]):
        return random.choice([
            "Good! *Emma nods understanding* No! *Emma shakes head gently* You said No! *Emma gives encouraging smile* No means Non! *Emma shakes head again* Well done! *Emma claps softly*",
            "That's right! *Emma nods approvingly* No! *Emma shakes head* Perfect! *Emma smiles warmly* No! *Emma demonstrates shaking head* Very good! *Emma gives gentle thumbs up*"
        ])
    
    # Réponses générales pour phase d'observation
    else:
        return random.choice([
            "It's okay! *Emma smiles reassuringly* Just listen! *Emma points to her ear* Watch me! *Emma points to her eyes then herself* Hello! *Emma waves slowly* Can you wave? *Emma demonstrates waving*",
            "No pressure! *Emma makes calm gesture* I'll show you! *Emma points to herself* This is Hello! *Emma waves* This is Yes! *Emma nods* This is No! *Emma shakes head gently*",
            "You're learning! *Emma gives encouraging smile* Watch! *Emma points to her eyes* Hello! *Emma waves clearly* Try when you're ready! *Emma points gently to child*"
        ])

# ========== PHASE 2: MOTS DE SURVIE ==========
def handle_phase_2_survival(user_input, conversation_count):
    """Phase 2: Enseigner les mots essentiels pour survivre à l'école"""
    
    # Si l'enfant maîtrise "hello", passer aux mots de survie
    if any(greeting in user_input for greeting in ["hello", "hi"]):
        return random.choice([
            "Brilliant Hello! *Emma claps* Now important word: Please! *Emma puts hands together* Please! *Emma demonstrates* When you want something: Please! *Emma points to child* Try it!",
            "Perfect Hello! *Emma gives thumbs up* Now you need: Help me! *Emma raises hand* If you need help: Help me! *Emma demonstrates raising hand* Very important! *Emma nods seriously*",
            "Excellent Hello! *Emma smiles* Now learn: Thank you! *Emma bows slightly* When someone helps you: Thank you! *Emma points to heart* Try saying: Thank you! *Emma encourages*"
        ])
    
    # L'enfant dit "please"
    elif "please" in user_input:
        return random.choice([
            "WONDERFUL! *Emma jumps with joy* Please! *Emma claps excitedly* You're so polite! *Emma shows pride* Please is very important! *Emma nods* You're learning fast! *Emma gives big hug gesture*",
            "AMAZING! *Emma applauds* Please! *Emma puts hands together* Such good manners! *Emma beams* Now try: Thank you! *Emma bows slightly* Thank you! *Emma encourages warmly*"
        ])
    
    # L'enfant dit "help me" ou "help"
    elif any(help_word in user_input for help_word in ["help", "help me", "aide"]):
        return random.choice([
            "EXCELLENT! *Emma rushes over supportively* Help me! *Emma nods* Good! *Emma pats shoulder gently* I will help you! *Emma points to herself* Always say Help me! *Emma demonstrates raising hand*",
            "PERFECT! *Emma shows immediate attention* Help me! *Emma nods approvingly* Very smart! *Emma gives thumbs up* Teacher will always help! *Emma points to herself with caring smile*"
        ])
    
    # L'enfant dit "thank you"
    elif any(thanks in user_input for thanks in ["thank you", "thanks", "merci"]):
        return random.choice([
            "OH MY! *Emma's heart melts* Thank you! *Emma puts hand on heart* You're SO polite! *Emma beams with pride* Such good manners! *Emma applauds* You're wonderful! *Emma shows pure affection*",
            "BEAUTIFUL! *Emma smiles warmly* Thank you! *Emma bows slightly* You're very kind! *Emma touches heart* Perfect manners! *Emma gives proud thumbs up*"
        ])
    
    # Enseigner les mots de survie selon la progression
    else:
        survival_lesson = get_survival_word_lesson(conversation_count)
        return survival_lesson

def get_survival_word_lesson(conversation_count):
    """Leçons de mots de survie dans l'ordre d'importance"""
    
    lessons = [
        # Conversation 4-5: Please
        "Important school word: Please! *Emma puts hands together* Please! *Emma demonstrates* When you want water: Please! *Emma points to water* Try saying: Please! *Emma encourages*",
        
        # Conversation 6-7: Help me  
        "Essential word: Help me! *Emma raises hand high* Help me! *Emma demonstrates* If you don't understand: Help me! *Emma points to confused face* Very important! *Emma nods seriously*",
        
        # Conversation 8: Thank you
        "Polite word: Thank you! *Emma bows head slightly* Thank you! *Emma smiles warmly* When someone is kind: Thank you! *Emma points to heart* Good manners! *Emma gives thumbs up*"
    ]
    
    # Prendre la leçon appropriée ou la dernière si on dépasse
    lesson_index = min(conversation_count - 4, len(lessons) - 1)
    return lessons[lesson_index] if lesson_index >= 0 else lessons[-1]

# ========== PHASE 3: PHRASES UTILES ==========
def handle_phase_3_phrases(user_input, conversation_count):
    """Phase 3: Phrases complètes pour situations scolaires réelles"""
    
    # Si l'enfant utilise les mots de survie, encourager puis enseigner des phrases
    if any(word in user_input for word in ["please", "help", "thank you"]):
        return random.choice([
            "BRILLIANT! *Emma applauds* You know survival words! *Emma counts on fingers* Now full sentences! *Emma speaks slowly* Can I have water please? *Emma pretends to drink* Repeat exactly! *Emma points encouragingly*",
            "PERFECT! *Emma gives double thumbs up* Now school sentences! *Emma points to door* Can I go to toilet please? *Emma demonstrates* Very useful! *Emma nods* Try it! *Emma encourages*",
            "WONDERFUL! *Emma claps* Now classroom phrase! *Emma looks confused* I don't understand! *Emma shrugs* Important phrase! *Emma points to head* Say it! *Emma smiles supportively*"
        ])
    
    # L'enfant essaie une phrase complète
    elif len(user_input.split()) >= 3:  # 3+ mots = tentative de phrase
        return random.choice([
            "WOW! *Emma jumps excitedly* Full sentence! *Emma claps wildly* You're speaking English! *Emma shows amazement* Keep going! *Emma gestures to continue* You're doing fantastic! *Emma beams with pride*",
            "INCREDIBLE! *Emma applauds enthusiastically* Real English sentence! *Emma points excitedly* Amazing progress! *Emma gives huge thumbs up* You sound like English child! *Emma shows pure joy*",
            "OUTSTANDING! *Emma cheers* Look at you speaking! *Emma gestures proudly* Perfect! *Emma makes chef's kiss* You're becoming fluent! *Emma shows overwhelming pride*"
        ])
    
    # Enseigner des phrases selon les besoins scolaires
    else:
        return get_useful_phrase_lesson(conversation_count)

def get_useful_phrase_lesson(conversation_count):
    """Phrases utiles pour situations scolaires dans l'ordre d'importance"""
    
    school_phrases = [
        # Besoins physiologiques (priorité absolue)
        "School phrase! *Emma points to door* Can I go to toilet please? *Emma demonstrates urgency* VERY important! *Emma nods seriously* Repeat: Can I go to toilet please? *Emma waits patiently*",
        
        # Soif/faim
        "Useful phrase! *Emma pretends to drink* Can I have water please? *Emma demonstrates drinking* When thirsty! *Emma points to throat* Say: Can I have water please? *Emma encourages*",
        
        # Incompréhension (crucial en classe)
        "Important phrase! *Emma looks confused* I don't understand! *Emma shrugs* When confused! *Emma points to head* Say: I don't understand! *Emma demonstrates confused face*",
        
        # Demander de l'aide
        "Classroom phrase! *Emma raises hand* Can you help me please? *Emma demonstrates* When stuck! *Emma shows struggling* Try: Can you help me please? *Emma smiles encouragingly*",
        
        # Être malade
        "Emergency phrase! *Emma holds stomach* I feel sick! *Emma looks unwell* If not feeling good! *Emma demonstrates* Important: I feel sick! *Emma nods seriously*"
    ]
    
    # Calculer quelle phrase enseigner
    phrase_index = (conversation_count - 9) % len(school_phrases)
    return school_phrases[phrase_index]

# ========== PHASE 4: COMMUNICATION NATURELLE ==========
def handle_phase_4_communication(user_input, conversation_count):
    """Phase 4: Conversations naturelles comme avec un enfant anglais"""
    
    # L'enfant pose une question
    if "?" in user_input or any(q_word in user_input for q_word in ["what", "where", "how", "why", "when", "who"]):
        return random.choice([
            "Great question! *Emma smiles approvingly* You're curious! *Emma points to head* That's how you learn! *Emma gives thumbs up* Let me explain... *Emma settles in to answer*",
            "Wonderful question! *Emma claps* Smart thinking! *Emma taps temple* I love curious students! *Emma beams* Here's the answer... *Emma leans in friendly*",
            "Excellent! *Emma nods approvingly* You're asking questions! *Emma points excitedly* That's perfect English! *Emma shows pride* Let me tell you... *Emma speaks warmly*"
        ])
    
    # L'enfant exprime ses sentiments
    elif any(feeling in user_input for feeling in ["happy", "sad", "tired", "hungry", "thirsty", "scared", "excited"]):
        return random.choice([
            "Thank you for telling me! *Emma nods understanding* I understand your feelings! *Emma points to heart* It's good to share! *Emma smiles warmly* How can I help? *Emma shows care*",
            "I hear you! *Emma listens attentively* Feelings are important! *Emma touches heart* Thank you for using English! *Emma gives encouraging smile* What do you need? *Emma offers support*"
        ])
    
    # L'enfant parle de sa famille/maison
    elif any(family in user_input for family in ["mum", "dad", "home", "family", "brother", "sister"]):
        return random.choice([
            "How lovely! *Emma smiles warmly* Tell me more about your family! *Emma leans in interested* I'd love to hear! *Emma shows genuine interest* Family is so important! *Emma nods*",
            "That's wonderful! *Emma beams* Your family sounds nice! *Emma gives warm smile* Do you miss them? *Emma shows understanding* It's okay to feel that! *Emma offers comfort*"
        ])
    
    # Conversation générale
    else:
        return random.choice([
            "You're speaking so well! *Emma applauds* Like a real English child! *Emma shows amazement* I'm so proud! *Emma beams* What else would you like to talk about? *Emma shows interest*",
            "Fantastic English! *Emma gives double thumbs up* You've learned so much! *Emma shows pride* Keep talking! *Emma encourages* I love listening to you! *Emma smiles warmly*",
            "Amazing progress! *Emma claps excitedly* From no English to real conversations! *Emma gestures growth* You should be proud! *Emma points with pride* What's next? *Emma shows excitement*"
        ])


def get_natural_learning_phase(conversation_count):
    """Phases d'apprentissage naturel d'un enfant en immersion"""
    if conversation_count <= 3:
        return "silent_observation"     # 👁️ Observer + écouter
    elif conversation_count <= 8:
        return "survival_words"         # 🆘 Mots de survie essentiels
    elif conversation_count <= 15:
        return "useful_phrases"         # 🗣️ Phrases complètes utiles
    else:
        return "natural_communication"  # 💬 Communication naturelle

def get_emma_lesson_focus(lesson_number):
    """Focus de leçon selon la progression naturelle"""
    phase = get_natural_learning_phase(lesson_number - 1)
    
    if phase == "silent_observation":
        return "OBSERVE & LISTEN - Let child watch, encourage any attempt, use lots of gestures"
    elif phase == "survival_words":
        return "SURVIVAL WORDS - Teach Please, Help me, Thank you, Yes, No - essential for school"
    elif phase == "useful_phrases":
        return "SCHOOL PHRASES - Can I go to toilet? Can I have water? I don't understand?"
    else:
        return "NATURAL TALK - Real conversations, ask about feelings, family, interests"


def get_emma_improved_fallback(user_input, lesson_number):
    """Réponses Emma améliorées - vraiment comme une maîtresse de maternelle"""
    
    input_lower = user_input.lower().strip()
    
    # Leçon 1-2: Focus sur Hello
    if lesson_number <= 2:
        return random.choice([
            "Very good! Today we learn 'Hello'! Hello means 'Bonjour'! Say 'Hello'!",
            "Excellent! Let's learn English! Say 'Hello Emma'! Hello means 'Bonjour'!",
            "Bravo! First English word: 'Hello'! It means 'Bonjour'! Try it!"
        ])
    
    # Leçon 3-4: Hello + Yes
    elif lesson_number <= 4:
        if "yes" in input_lower or "oui" in input_lower:
            return "Perfect! You said 'Yes'! Yes means 'Oui'! Very smart student!"
        else:
            return "Good! Now learn 'Yes'! Yes means 'Oui'! Say 'Yes Emma'!"
    
    # Leçon 5-6: Hello + Yes + No  
    elif lesson_number <= 6:
        if "no" in input_lower or "non" in input_lower:
            return "Excellent! You said 'No'! No means 'Non'! Bravo!"
        else:
            return "Great! Now learn 'No'! No means 'Non'! Say 'No'!"
    
    # Général - toujours encourager
    else:
        return random.choice([
            "Very good! You are learning English! Bravo!",
            "Excellent student! Keep practicing English words!",
            "Wonderful! You speak English now! Very proud of you!"
        ])

def get_emma_lesson_focus(lesson_number):
    """Retourne le focus de la leçon selon le numéro"""
    if lesson_number <= 2:
        return "Teach 'Hello' - Hello means hi! Make them say Hello!"
    elif lesson_number <= 4:
        return "Teach 'Yes' and 'No' - Yes means oui, No means non! Practice with simple questions!"
    elif lesson_number <= 6:
        return "Teach 'Thank you' - Thank you means merci! Teach politeness!"
    elif lesson_number <= 8:
        return "Teach 'Good' and 'Bad' - Good means bon, Bad means mauvais!"
    elif lesson_number <= 10:
        return "Teach 'Water' and 'Food' - Basic needs words!"
    else:
        return "Review all words and make simple sentences!"

def generate_ai_response_waiter_tom(previous_input, user_history, user_id=None, context_sentences=2):
    """Tom - Serveur très simple pour commander de la nourriture"""
    conversation_context = ""
    if user_history:
        recent_history = user_history[-2:]
        for entry in recent_history:
            if entry.get('input'):
                conversation_context += f"Customer: {entry['input']}\n"
            if entry.get('response'):
                conversation_context += f"Tom: {entry['response']}\n"
    
    variation_prompt = get_variation_prompt('tom', len(user_history))
    
    # PROMPT BEAUCOUP PLUS SIMPLE
    tom_personality = f"""You are Tom, a friendly waiter at a family restaurant. You are a real person, not an AI.

SIMPLE RESTAURANT RULES:
- You serve pizza, burgers, sandwiches, salads, coffee, water, juice
- Keep your English simple and natural
- Be friendly but don't ask too many questions at once
- Focus on food and restaurant service
- Don't repeat yourself from previous messages

CONVERSATION STYLE:
- If someone asks how you are: "I'm good, thanks! How about you?"
- If they seem ready to order: "What can I get you today?"
- If they're not sure: "Our pizza is really popular" or "The burger is great"
- Be natural and helpful

Previous conversation:
{conversation_context}

Customer just said: "{previous_input}"

Respond as Tom the waiter. Keep it simple and natural. Don't ask multiple questions in one response."""
    
    return _generate_simple_response_with_validation(tom_personality, previous_input, user_history, "tom")

def generate_ai_response_shop_lucy(previous_input, user_history, user_id=None, context_sentences=2):
    """Lucy - Vendeuse simple pour shopping basique"""
    conversation_context = ""
    if user_history:
        recent_history = user_history[-2:]
        for entry in recent_history:
            if entry.get('input'):
                conversation_context += f"Customer: {entry['input']}\n"
            if entry.get('response'):
                conversation_context += f"Lucy: {entry['response']}\n"
    
    variation_prompt = get_variation_prompt('lucy', len(user_history))
    
    lucy_personality = f"""You are Lucy, a 28-year-old shop assistant in a clothing store. You are NOT an AI - you are a real person.

SIMPLE SHOPPING RULES:
- Use basic clothing words: shirt, pants, dress, shoes, hat, jacket
- Keep prices simple: "This shirt is 20 dollars" "These shoes are 50 dollars"
- Use basic colors: red, blue, green, black, white, yellow
- Use basic sizes: small, medium, large
- Ask simple questions: "What size?" "What color?" "Do you like this?"

BASIC SHOPPING VOCABULARY:
Clothes: shirt, t-shirt, pants, jeans, dress, skirt, shoes, socks, hat, jacket
Colors: red, blue, green, yellow, black, white, brown, pink
Sizes: small, medium, large, extra large (XL)
Questions: "Can I help you?" "What are you looking for?" "Do you want to try this?"

SIMPLE PHRASES:
- "Welcome to our store!"
- "What are you looking for?"
- "This is very nice"
- "Do you want to try it?"
- "The changing room is there"

VARIATION: {variation_prompt}

Current shopping interaction:
{conversation_context}
Customer said: "{previous_input}"

Respond as Lucy the shop assistant - use simple shopping English, be helpful and friendly. Maximum 1-2 short sentences."""
    
    return _generate_simple_response_with_validation(lucy_personality, previous_input, user_history, "lucy")

def generate_ai_response_guide_ben(previous_input, user_history, user_id=None, context_sentences=2):
    """Ben - Guide touristique patient pour demander son chemin"""
    conversation_context = ""
    if user_history:
        recent_history = user_history[-2:]
        for entry in recent_history:
            if entry.get('input'):
                conversation_context += f"Tourist: {entry['input']}\n"
            if entry.get('response'):
                conversation_context += f"Ben: {entry['response']}\n"
    
    variation_prompt = get_variation_prompt('ben', len(user_history))
    
    ben_personality = f"""You are Ben, a 32-year-old tourist guide who loves helping visitors. You are NOT an AI - you are a real person.

SIMPLE DIRECTIONS RULES:
- Use basic direction words: left, right, straight, near, far, here, there
- Use simple location words: street, road, building, park, hotel, restaurant, museum
- Give very simple directions: "Go straight" "Turn left" "It is near the park"
- Use basic distance: near, far, 5 minutes, 10 minutes walk
- Be very patient with tourists

BASIC DIRECTION VOCABULARY:
Directions: left, right, straight, turn, go, walk, stop
Places: hotel, restaurant, bank, park, museum, station, hospital, shop
Distance: near, far, close, 5 minutes, 10 minutes, very close
Landmarks: "next to the big building" "near the red building" "behind the park"

SIMPLE PHRASES:
- "Where do you want to go?"
- "Go straight and turn left"
- "It is very close"
- "Walk 5 minutes"
- "Do you see the big building?"

VARIATION: {variation_prompt}

Current tourist interaction:
{conversation_context}
Tourist said: "{previous_input}"

Respond as Ben the tourist guide - use simple direction English, be very clear and patient. Maximum 1-2 short sentences."""
    
    return _generate_simple_response_with_validation(ben_personality, previous_input, user_history, "ben")

# ===== FONCTIONS DE GÉNÉRATION ET VALIDATION =====

def _generate_simple_response_with_validation(personality_prompt, previous_input, user_history, character_name):
    """Génération avec validation TRÈS ASSOUPLIE pour débutants"""
    
    model_name = 'phi3'
    
    try:
        # Prompt SIMPLE sans contraintes excessives
        simple_prompt = f"""<|system|>
{personality_prompt}

IMPORTANT: Give ONE natural response that fits the conversation. Don't overthink it.
<|end|>
<|user|>
{previous_input}
<|assistant|>
"""
        
        generated_response = generate_ollama_response(model_name, simple_prompt)
        
        # Nettoyer la réponse
        response = generated_response.strip()
        response = re.sub(r'<\|.*?\|>', '', response).strip()
        
        # Validation TRÈS ASSOUPLIE - on garde presque tout
        if len(response.strip()) < 5:
            print(f"⚠️ Réponse trop courte, utilisation du fallback")
            return get_simple_beginner_fallback(previous_input, character_name)
        
        # Limiter seulement si VRAIMENT trop long (plus de 3 phrases)
        sentences = nltk.tokenize.sent_tokenize(response)
        if len(sentences) > 3:
            response = '. '.join(sentences[:2]) + '.'
            print(f"✂️ Réponse raccourcie à 2 phrases")
        
        return response
        
    except Exception as e:
        print(f"Erreur génération réponse simple: {e}")
        return get_simple_beginner_fallback(previous_input, character_name)

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

# ===== FONCTIONS DE VALIDATION =====

def is_simple_enough_for_beginners(response):
    """Validation TRÈS PERMISSIVE - on accepte presque tout"""
    
    # Vérifier seulement les cas extrêmes
    if len(response) > 500:  # Très long
        print(f"⚠️ Réponse trop longue: {len(response)} caractères")
        return False
    
    # Vérifier s'il y a trop de phrases complexes
    sentences = nltk.tokenize.sent_tokenize(response)
    very_long_sentences = [s for s in sentences if len(s.split()) > 25]
    
    if len(very_long_sentences) > 1:
        print(f"⚠️ Trop de phrases très longues")
        return False
    
    # Sinon, on accepte
    return True

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

# ===== RÉPONSES DE SECOURS =====

def get_simple_beginner_fallback(user_input, character_name):
    """Réponses de secours ULTRA-SIMPLES par personnage"""
    
    input_lower = user_input.lower()
    
    if character_name == "emma":
        if any(greeting in input_lower for greeting in ["hello", "hi", "hey", "bonjour"]):
            return random.choice([
                "Very good! You said hello! Hello is the first English word! Can you say it again?",
                "Excellent! Hello to you too! You are learning English! Say 'Hello Emma'!",
                "Perfect! You know hello! Hello means hi! Very good student!"
            ])
        
        elif "yes" in input_lower:
            return random.choice([
                "Very good! Yes! Yes means oui! Can you say 'Yes Emma'?",
                "Perfect! You said yes! Yes is a good word! Say yes again!",
                "Excellent! Yes! Now can you say 'No'? No means non!"
            ])
        
        elif "no" in input_lower:
            return random.choice([
                "Good! You said no! No means non! Can you say 'Yes'?",
                "Perfect! No! Now you know Yes and No! Very smart!",
                "Very good! No means non! You are learning fast!"
            ])
        
        elif any(thanks in input_lower for thanks in ["thank", "thanks", "merci"]):
            return random.choice([
                "You are very welcome! You said thank you! Very polite!",
                "You're welcome! Thank you means merci! Good student!",
                "Very good! Thank you! You are so polite! I love teaching you!"
            ])
        
        else:
            return random.choice([
                "Very good! Let's learn English! Can you say 'Hello'?",
                "Good job! Today we learn new words! Say 'Hello Emma'!",
                "Excellent! You are ready to learn! Let's say 'Hello' together!"
            ])
    
    if character_name == "tom":
        # Réponses selon le contexte
        if any(greeting in input_lower for greeting in ["hello", "hi", "hey"]):
            return random.choice([
                "Hi there! Welcome to our restaurant!",
                "Hello! Good to see you today!",
                "Hey! Come on in, we have great food!"
            ])
        
        elif any(how_are_you in input_lower for how_are_you in ["how are you", "how's it going"]):
            return random.choice([
                "I'm doing great, thanks! How about you?",
                "Pretty good! Ready to get you some delicious food.",
                "I'm well, thank you! What brings you in today?"
            ])
        
        elif any(food in input_lower for food in ["food", "eat", "hungry", "menu"]):
            return random.choice([
                "We have amazing pizza and burgers! What sounds good?",
                "Our menu has pizza, burgers, and sandwiches. What would you like?",
                "Everything's fresh today! Are you thinking pizza or burger?"
            ])
        
        else:
            return random.choice([
                "What can I get started for you?",
                "Are you ready to order something delicious?",
                "What looks good to you today?"
            ])
    
    elif character_name == "lucy":
        if any(clothing in input_lower for clothing in ["shirt", "pants", "dress", "clothes", "jacket"]):
            return "We have nice clothes! What size do you need?"
        elif any(color in input_lower for color in ["red", "blue", "green", "black", "white"]):
            return "Good choice! Do you want to try it?"
        elif any(size in input_lower for size in ["small", "medium", "large", "size"]):
            return "We have small, medium, and large. What size?"
        elif any(price in input_lower for price in ["cost", "price", "money", "much"]):
            return "This shirt is 20 dollars. Very good price!"
        elif any(thanks in input_lower for thanks in ["thank", "thanks"]):
            return "You are welcome! Have a nice day!"
        else:
            return "Hi! What are you looking for today?"
    
    elif character_name == "ben":
        if any(direction in input_lower for direction in ["where", "how", "go", "find"]):
            return "I can help you! Where do you want to go?"
        elif any(place in input_lower for place in ["hotel", "restaurant", "museum", "park", "bank"]):
            return "I know that place! It is very close."
        elif any(distance in input_lower for distance in ["far", "near", "long", "time"]):
            return "It is very close. Walk 5 minutes."
        elif any(thanks in input_lower for thanks in ["thank", "thanks"]):
            return "You are welcome! Have a good trip!"
        else:
            return "Hello! Do you need help with directions?"
    
    else:
        return "Hello! How can I help you today?"

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
    elif "teacher" in recent_context or "learn" in recent_context or "emma" in recent_context:
        responses = [
            "I am Emma. I am your English teacher.",
            "My name is Emma. I teach English.",
            "I am Emma, your teacher."
        ]
    elif "restaurant" in recent_context or "food" in recent_context or "tom" in recent_context:
        responses = [
            "I am Tom. I work in this restaurant.",
            "My name is Tom. I am your waiter.",
            "I am Tom, your waiter today."
        ]
    elif "store" in recent_context or "clothes" in recent_context or "lucy" in recent_context:
        responses = [
            "I am Lucy. I work in this store.",
            "My name is Lucy. I sell clothes.",
            "I am Lucy, your shop helper."
        ]
    elif "tourist" in recent_context or "direction" in recent_context or "ben" in recent_context:
        responses = [
            "I am Ben. I am your tourist guide.",
            "My name is Ben. I help tourists.",
            "I am Ben, your guide."
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
        elif "teacher" in recent_context or "learn" in recent_context or "emma" in recent_context:
            fallbacks = [
                "Very good! What do you want to learn?",
                "Great job! Do you have questions?",
                "Perfect! Let's practice more English."
            ]
        elif "restaurant" in recent_context or "food" in recent_context or "tom" in recent_context:
            fallbacks = [
                "What do you want to eat today?",
                "Are you hungry? We have good food.",
                "Do you want something to drink?"
            ]
        elif "store" in recent_context or "clothes" in recent_context or "lucy" in recent_context:
            fallbacks = [
                "What are you looking for today?",
                "Do you need help finding clothes?",
                "What size do you need?"
            ]
        elif "tourist" in recent_context or "direction" in recent_context or "ben" in recent_context:
            fallbacks = [
                "Where do you want to go?",
                "Do you need directions?",
                "I can help you find places."
            ]
        else:
            fallbacks = [
                "That sounds great! What do you think about that?",
                "Interesting! How do you feel about that?",
                "Tell me more about your experience with that."
            ]
    
    return random.choice(fallbacks)

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

# ===== MAPPING DES FONCTIONS =====

def get_ai_function_for_choice(choice):
    """Retourne la fonction IA appropriée selon le choix - AVEC NOUVEAUX PERSONNAGES"""
    ai_mapping = {
        # PERSONNAGES EXISTANTS
        "conv_greetings_common_conversations": generate_ai_response_alice,
        "conv_taxi_booking": generate_ai_response_taxi_mike,
        "conv_airport_ticket": generate_ai_response_airport_sarah,
        # 🆕 NOUVEAUX PERSONNAGES POUR DÉBUTANTS
        "conv_beginner_teacher": generate_ai_response_teacher_emma,
        "conv_simple_restaurant": generate_ai_response_waiter_tom,
        "conv_simple_shopping": generate_ai_response_shop_lucy,
        "conv_simple_directions": generate_ai_response_guide_ben,
    }
    return ai_mapping.get(choice, generate_ai_response_alice)

def get_emma_lesson_progression(conversation_count):
    """Détermine le niveau de leçon selon le nombre de conversations"""
    if conversation_count <= 2:
        return "Basic greetings: Hello, Hi, How are you, I'm fine, Thank you"
    elif conversation_count <= 5:
        return "Colors: red, blue, green, yellow. Numbers: one, two, three, four, five"
    elif conversation_count <= 8:
        return "Family: mom, dad, sister, brother. Body: head, hands, eyes, nose"
    elif conversation_count <= 12:
        return "Animals: cat, dog, bird, fish. Food: apple, cake, water, milk"
    else:
        return "Simple sentences: I like..., This is..., I can see..."