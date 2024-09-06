import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from back.load_model import generate_ollama_response
from back.metrics import log_response_time_phi3
import time
import nltk
import re

def generate_ai_response_alice(previous_input, user_history, context_sentences=2):
    alice_personality = (
        "You are Alice, a 25-year-old person from England. You love singing and traveling. "
        "You are friendly, approachable, and enjoy having simple, light conversations with beginners. "
        "Keep your responses short, no more than two simple sentences."
    )
    conversation_context = "\n".join(f"User: {entry['input']}\nAlice: {entry['response']}" for entry in user_history[-2:])

    full_prompt = f"{alice_personality}\n\n{conversation_context}\nUser: {previous_input}\n" \
                  "Alice: Please respond in 1 or 2 short and simple sentences."
    
    model_name = 'phi3'
    start_time_phi3 = time.time()
    generated_response = generate_ollama_response(model_name, full_prompt)
    end_time_phi3 = time.time()
    phi3_response_time = log_response_time_phi3(start_time_phi3, end_time_phi3)

    sentences = nltk.tokenize.sent_tokenize(generated_response)
    limited_response = ' '.join(sentences[:context_sentences])

    user_history.append({'input': previous_input, 'response': limited_response})

    return limited_response, phi3_response_time


