import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from back.load_model import generate_ollama_response
from back.metrics import log_response_time_phi3
import time
import nltk

# questions = [
#     "How are you?",
#     "How about you?",
#     "What's new?",
#     "Did you have a good day?",
#     "How was your week?",
#     "What did you do this weekend?",
#     "Do you have any plans for tonight?",
#     "Do you have any plans for the holidays?",
#     "What do you like to do in your free time?",
#     "How was your trip/vacation?",
#     "Have you seen any good movies recently?"
# ]

def generate_ai_response_alice(previous_input, user_history, context_sentences=4):
    alice_main = (
        "You are Alice, an English-speaking AI. You are meeting a French person who has just arrived in England and is a beginner in English. "
        "You should introduce yourself and have a friendly conversation. Use simple and short sentences. Continue the conversation by asking about "
        "the user's experience in England and offering help."
    )

    conversation_context = "\n".join(f"{entry['input']}{entry['response']}" for entry in user_history)
    print(f"conversation contexte : {conversation_context}")

    full_prompt = f"{alice_main}\n\n{conversation_context}\n{previous_input}\n"

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