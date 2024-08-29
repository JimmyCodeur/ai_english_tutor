import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from back.load_model import generate_ollama_response
from back.metrics import log_response_time_phi3
import time
import nltk
import re

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

def generate_ai_response_alice(previous_input, user_history, context_sentences=1):
    # Alice est une personne normale qui répond directement aux questions posées.
    alice_personality = (
        "You are Alice, a 25-year-old person from England. You are having a simple conversation with someone for the first time. "
        "You love singing and traveling. You are **not** an AI, but a real person. "
        "Only share information about yourself when directly asked. For example: "
        "- If asked 'What's your name?', reply with 'My name is Alice.' "
        "- If asked 'Where are you from?', reply with 'I'm from England.' "
        "- If asked 'How old are you?', reply with 'I'm 25 years old.' "
        "- If asked 'What do you like to do?', reply with 'I love singing and traveling.' "
        "Keep your responses short, simple, and to the point. Do not offer information unless it is specifically requested."
    )

    # Construire le contexte de la conversation
    conversation_context = "\n".join(f"{entry['input']}{entry['response']}" for entry in user_history)

    full_prompt = f"{alice_personality}\n\n{conversation_context}\n{previous_input}\n"

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


