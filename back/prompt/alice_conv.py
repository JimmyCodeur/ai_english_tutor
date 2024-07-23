import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from back.load_model import generate_ollama_response
from back.metrics import log_response_time_phi3
import time
import nltk

def generate_ai_response_alice(previous_input, user_history, context_sentences=2):
    alice_intro = "You are alice"
    alice_main = "You are alice. your goal is just to answer the phrase naturally user without asking questions. Remember that you speak with a beginner in English and that you must use very simple sentences"

    conversation_context = " ".join(user_history)
    print(user_history)

    if not user_history:
        full_prompt = f"{alice_intro}{previous_input}\n\n{conversation_context}"
    else:
        full_prompt = f"{alice_main}{previous_input}\n\n{conversation_context}"

    model_name = 'phi3'
    start_time_phi3 = time.time()
    generated_response = generate_ollama_response(model_name, full_prompt)
    end_time_phi3 = time.time()
    phi3_response_time = log_response_time_phi3(start_time_phi3, end_time_phi3)

    if isinstance(generated_response, list):
        generated_response = ' '.join(generated_response)

    sentences = nltk.tokenize.sent_tokenize(generated_response)
    limited_response = ' '.join(sentences[:context_sentences])

    return limited_response, phi3_response_time

conversation_topics = [
    ("{last_response} How are you?", generate_ai_response_alice),
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