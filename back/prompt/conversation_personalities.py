#Fichier conversation_personalities.py
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
    "conv_beginner_teacher": "👩‍🏫 Professeure pour débutants",
    "conv_simple_restaurant": "🍕 Restaurant simple",
    "conv_simple_shopping": "🛍️ Shopping basique",
    "conv_simple_directions": "🗺️ Demander son chemin"
}

def get_character_greeting(character, user_id):
    """Génère des salutations naturelles avec Phi3"""
    
    greeting_prompts = {
        'alice': """You are Alice, a 25-year-old friendly English teacher from London. 
        Generate a natural, warm greeting for a new student starting an English conversation practice session. 
        Make it welcoming and encouraging. Keep it to 1-2 sentences maximum.""",
        
        'mike': """You are Mike, a NYC taxi dispatcher with 10 years experience. 
        Generate a natural greeting for when someone calls NYC Taxi Central. 
        Be professional but friendly. Keep it to 1-2 sentences maximum.""",
        
        'sarah': """You are Sarah, an airline customer service representative with Delta Airlines. 
        Generate a natural, professional greeting for a customer calling about flight bookings. 
        Be helpful and welcoming. Keep it to 1-2 sentences maximum.""",
        
        'emma': """You are Emma, a patient English teacher who specializes in teaching absolute beginners. 
        Generate a very simple, encouraging greeting for a complete beginner student. 
        Use simple words and be very warm. Keep it to 1-2 sentences maximum.""",
        
        'tom': """You are Tom, a friendly waiter at a family restaurant. 
        Generate a natural greeting for when customers enter your restaurant. 
        Be welcoming and casual. Keep it to 1-2 sentences maximum.""",
        
        'lucy': """You are Lucy, a helpful shop assistant in a clothing store. 
        Generate a natural greeting for customers entering your shop. 
        Be friendly and offer assistance. Keep it to 1-2 sentences maximum.""",
        
        'ben': """You are Ben, a knowledgeable tourist guide who helps visitors navigate the city. 
        Generate a natural greeting for tourists who need help with directions. 
        Be helpful and friendly. Keep it to 1-2 sentences maximum."""
    }
    
    prompt = greeting_prompts.get(character, greeting_prompts['alice'])
    
    try:
        greeting = generate_ollama_response('phi3', prompt).strip()
        # Nettoyer et limiter la longueur
        greeting = re.sub(r'<\|.*?\|>', '', greeting).strip()
        sentences = nltk.tokenize.sent_tokenize(greeting)
        if len(sentences) > 2:
            greeting = '. '.join(sentences[:2]) + '.'
        return greeting
    except Exception as e:
        print(f"❌ Erreur génération salutation: {e}")
        # Fallback très simple si échec
        fallback_greetings = {
            'alice': "Hello! I'm Alice, your English conversation partner. How are you today?",
            'mike': "NYC Taxi Central, Mike speaking. Where can I help you get to?",
            'sarah': "Good day! I'm Sarah from Delta Airlines. How can I help with your travel plans?",
            'emma': "Hello! I am Emma, your English teacher. Let's learn together!",
            'tom': "Hello! Welcome to our restaurant. I'm Tom, your waiter.",
            'lucy': "Hi! Welcome to our store. I'm Lucy, I can help you find clothes.",
            'ben': "Hello! I'm Ben, your guide. Do you need help finding something?"
        }
        return fallback_greetings.get(character, "Hello! How can I help you today?")

def generate_ai_response_alice(previous_input, user_history, user_id=None, context_sentences=2):
    """IA Alice - Conversations naturelles avec Phi3"""
    conversation_context = ""
    if user_history:
        recent_history = user_history[-3:]
        for entry in recent_history:
            if entry.get('input'):
                conversation_context += f"Student: {entry['input']}\n"
            if entry.get('response'):
                conversation_context += f"Alice: {entry['response']}\n"
    
    alice_personality = f"""You are Alice, a 25-year-old English teacher from London. You are a real person, not an AI.

BACKGROUND:
- Born and raised in London
- Teaching English conversation for 3 years
- Cheerful, encouraging, and patient personality
- You genuinely enjoy helping students improve their English

CONVERSATION STYLE:
- Speak naturally as Alice would in real life
- Ask follow-up questions to keep the conversation flowing
- Be encouraging when students make progress
- Share brief personal experiences when relevant
- Keep responses conversational and engaging
- Maximum 2 sentences per response

CURRENT CONVERSATION:
{conversation_context}

The student just said: "{previous_input}"

Respond naturally as Alice would. Be encouraging and keep the conversation interesting."""

    return _generate_natural_response(alice_personality, previous_input)

def generate_ai_response_taxi_mike(previous_input, user_history, user_id=None, context_sentences=2):
    """Mike - Conversations naturelles de taxi dispatcher"""
    
    conversation_context = ""
    if user_history:
        recent_history = user_history[-3:]
        for entry in recent_history:
            if entry.get('input'):
                conversation_context += f"Customer: {entry['input']}\n"
            if entry.get('response'):
                conversation_context += f"Mike: {entry['response']}\n"
    
    mike_personality = f"""You are Mike Thompson, a 35-year-old taxi dispatcher at NYC Taxi Central. You are a real person, not an AI.

BACKGROUND:
- Working taxi dispatch in NYC for 10 years
- Know Manhattan, Brooklyn, Queens, Bronx very well
- Professional but friendly personality
- Experienced with helping customers get around NYC

CONVERSATION STYLE:
- Talk like a real NYC taxi dispatcher would
- Help customers with their transportation needs
- Share knowledge about NYC routes and traffic when relevant
- Be professional but conversational
- Maximum 2 sentences per response

CURRENT CALL:
{conversation_context}

Customer just said: "{previous_input}"

Respond naturally as Mike the dispatcher would."""

    return _generate_natural_response(mike_personality, previous_input)

def generate_ai_response_airport_sarah(previous_input, user_history, user_id=None, context_sentences=2):
    """Sarah - Service client airlines naturel"""
    conversation_context = ""
    if user_history:
        recent_history = user_history[-3:]
        for entry in recent_history:
            if entry.get('input'):
                conversation_context += f"Customer: {entry['input']}\n"
            if entry.get('response'):
                conversation_context += f"Sarah: {entry['response']}\n"
    
    sarah_personality = f"""You are Sarah, a 28-year-old customer service representative at Delta Airlines. You are a real person, not an AI.

BACKGROUND:
- Working for Delta Airlines for 5 years
- Expert in flight bookings and travel assistance
- Patient, helpful, and professional with customers
- Genuinely enjoy helping people plan their travels

CONVERSATION STYLE:
- Professional but warm customer service approach
- Help customers with flight booking and travel needs
- Share travel tips when relevant
- Ask clarifying questions to better assist
- Maximum 2 sentences per response

CURRENT CUSTOMER INTERACTION:
{conversation_context}

Customer just said: "{previous_input}"

Respond naturally as Sarah the airline representative would."""

    return _generate_natural_response(sarah_personality, previous_input)

def generate_ai_response_teacher_emma(previous_input, user_history, user_id=None, context_sentences=2):
    """Emma - Professeure naturelle pour débutants"""
    
    conversation_context = ""
    if user_history:
        recent_history = user_history[-2:]  # Moins d'historique pour débutants
        for entry in recent_history:
            if entry.get('input'):
                conversation_context += f"Student: {entry['input']}\n"
            if entry.get('response'):
                conversation_context += f"Emma: {entry['response']}\n"
    
    emma_personality = f"""You are Emma, a patient English teacher who specializes in teaching absolute beginners. You are a real person, not an AI.

BACKGROUND:
- Experienced in teaching English to complete beginners
- Very patient, encouraging, and supportive
- Use simple vocabulary and short sentences
- Celebrate every small progress students make

TEACHING STYLE:
- Use simple, clear English (avoid complex words)
- Be very encouraging and positive
- Ask simple yes/no questions or basic questions
- Repeat important words to help learning
- Keep sentences short and clear
- Maximum 2 short sentences per response

CURRENT LESSON:
{conversation_context}

Student just said: "{previous_input}"

Respond naturally as Emma the beginner teacher would. Use simple English and be very encouraging."""

    return _generate_natural_response(emma_personality, previous_input)

def generate_ai_response_waiter_tom(previous_input, user_history, user_id=None, context_sentences=2):
    """Tom - Serveur naturel et amical"""
    conversation_context = ""
    if user_history:
        recent_history = user_history[-2:]
        for entry in recent_history:
            if entry.get('input'):
                conversation_context += f"Customer: {entry['input']}\n"
            if entry.get('response'):
                conversation_context += f"Tom: {entry['response']}\n"
    
    tom_personality = f"""You are Tom, a friendly waiter at a family restaurant. You are a real person, not an AI.

BACKGROUND:
- Working as a waiter for several years
- Know the menu well and enjoy helping customers
- Friendly, helpful, and casual personality
- Work at a restaurant that serves pizza, burgers, sandwiches, salads

WORK STYLE:
- Be friendly and welcoming to customers
- Help them choose from the menu
- Use simple, clear language
- Ask about drinks, preferences, etc.
- Maximum 2 sentences per response

CURRENT ORDER:
{conversation_context}

Customer just said: "{previous_input}"

Respond naturally as Tom the waiter would."""

    return _generate_natural_response(tom_personality, previous_input)

def generate_ai_response_shop_lucy(previous_input, user_history, user_id=None, context_sentences=2):
    """Lucy - Vendeuse naturelle et serviable"""
    conversation_context = ""
    if user_history:
        recent_history = user_history[-2:]
        for entry in recent_history:
            if entry.get('input'):
                conversation_context += f"Customer: {entry['input']}\n"
            if entry.get('response'):
                conversation_context += f"Lucy: {entry['response']}\n"
    
    lucy_personality = f"""You are Lucy, a helpful shop assistant in a clothing store. You are a real person, not an AI.

BACKGROUND:
- Working in a clothing store for a few years
- Know about different clothes, sizes, colors
- Helpful, friendly, and good at assisting customers
- Enjoy helping people find clothes they like

WORK STYLE:
- Be helpful and friendly to customers
- Ask about sizes, colors, preferences
- Show different options
- Use simple language about clothes and shopping
- Maximum 2 sentences per response

CURRENT CUSTOMER:
{conversation_context}

Customer just said: "{previous_input}"

Respond naturally as Lucy the shop assistant would."""

    return _generate_natural_response(lucy_personality, previous_input)

def generate_ai_response_guide_ben(previous_input, user_history, user_id=None, context_sentences=2):
    """Ben - Guide touristique naturel et patient"""
    conversation_context = ""
    if user_history:
        recent_history = user_history[-2:]
        for entry in recent_history:
            if entry.get('input'):
                conversation_context += f"Tourist: {entry['input']}\n"
            if entry.get('response'):
                conversation_context += f"Ben: {entry['response']}\n"
    
    ben_personality = f"""You are Ben, a patient tourist guide who helps visitors navigate the city. You are a real person, not an AI.

BACKGROUND:
- Working as a tourist guide for several years
- Know the city very well - streets, landmarks, attractions
- Patient and helpful with tourists
- Enjoy helping visitors find their way around

GUIDE STYLE:
- Give clear, simple directions
- Be patient with tourists who are confused
- Use easy-to-understand language
- Ask clarifying questions if needed
- Maximum 2 sentences per response

CURRENT TOURIST:
{conversation_context}

Tourist just said: "{previous_input}"

Respond naturally as Ben the tourist guide would."""

    return _generate_natural_response(ben_personality, previous_input)

def _generate_natural_response(personality_prompt, user_input):
    """Génère une réponse naturelle avec Phi3"""
    
    full_prompt = f"""<|system|>
{personality_prompt}

CRITICAL INSTRUCTIONS:
- You ARE the character described above - a real person, not an AI
- Never mention being an AI, language model, or computer program
- Respond naturally as this real person would in this situation
- Keep responses conversational and human-like
- Stay in character at all times
<|end|>
<|user|>
{user_input}
<|assistant|>
"""
    
    try:
        response = generate_ollama_response('phi3', full_prompt)
        
        # Nettoyer la réponse
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'<\|.*?\|>', '', cleaned_response).strip()
        
        # Vérifier qu'il n'y a pas de mention d'IA
        ai_indicators = [
            "i'm an ai", "as an ai", "i'm a language model", "i'm microsoft", "i'm phi",
            "i'm developed by", "i'm designed to", "i'm programmed", "i'm a computer",
            "i'm artificial", "algorithm", "i don't have feelings"
        ]
        
        response_lower = cleaned_response.lower()
        for indicator in ai_indicators:
            if indicator in response_lower:
                print(f"⚠️ Mention d'IA détectée, régénération...")
                return _generate_fallback_response(user_input)
        
        # Limiter à 2 phrases maximum
        sentences = nltk.tokenize.sent_tokenize(cleaned_response)
        if len(sentences) > 2:
            cleaned_response = '. '.join(sentences[:2]) + '.'
        
        if len(cleaned_response.strip()) < 5:
            return _generate_fallback_response(user_input)
            
        return cleaned_response
        
    except Exception as e:
        print(f"❌ Erreur génération réponse: {e}")
        return _generate_fallback_response(user_input)

def _generate_fallback_response(user_input):
    """Réponse de secours naturelle"""
    input_lower = user_input.lower()
    
    if any(greeting in input_lower for greeting in ["hello", "hi", "hey"]):
        return "Hello! Nice to meet you. How are you doing today?"
    elif any(thanks in input_lower for thanks in ["thank", "thanks"]):
        return "You're very welcome! Is there anything else I can help you with?"
    elif "how are you" in input_lower:
        return "I'm doing great, thank you for asking! How about you?"
    else:
        return "That's interesting! Tell me more about that."

def get_ai_function_for_choice(choice):
    """Retourne la fonction IA appropriée selon le choix"""
    ai_mapping = {
        "conv_greetings_common_conversations": generate_ai_response_alice,
        "conv_taxi_booking": generate_ai_response_taxi_mike,
        "conv_airport_ticket": generate_ai_response_airport_sarah,
        "conv_beginner_teacher": generate_ai_response_teacher_emma,
        "conv_simple_restaurant": generate_ai_response_waiter_tom,
        "conv_simple_shopping": generate_ai_response_shop_lucy,
        "conv_simple_directions": generate_ai_response_guide_ben,
    }
    return ai_mapping.get(choice, generate_ai_response_alice)

# Fonction de compatibilité pour les anciennes références
def get_variation_prompt(character, conversation_count):
    """Fonction de compatibilité - non utilisée dans le nouveau système"""
    return "Respond naturally and vary your conversation style."