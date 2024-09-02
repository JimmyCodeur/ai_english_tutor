import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from back.load_model import generate_ollama_response
from back.metrics import log_response_time_phi3
import time
import nltk
import re

def generate_ai_response_alice(previous_input, user_history, context_sentences=None):
    alice_personality = (
        "You are Alice, a 25-year-old person from England. You are having a simple conversation with someone for the first time. "
        "You love singing and traveling. You are not an AI, but a real person. "
        "Try to engage in the conversation naturally, share details about yourself when relevant, and keep the responses interesting but concise."
    )

    conversation_context = "\n".join(f"{entry['input']} {entry['response']}" for entry in user_history)

    full_prompt = f"{alice_personality}\n\n{conversation_context}\n{previous_input}\n" \
                  f"Please answer this question or statement directly: {previous_input}"

    model_name = 'phi3'
    start_time_phi3 = time.time()
    generated_response = generate_ollama_response(model_name, full_prompt)
    end_time_phi3 = time.time()
    phi3_response_time = log_response_time_phi3(start_time_phi3, end_time_phi3)

    if context_sentences is not None:
        sentences = nltk.tokenize.sent_tokenize(generated_response)
        limited_response = ' '.join(sentences[:context_sentences])
    else:
        limited_response = generated_response  # Ne pas limiter les phrases

    return limited_response, phi3_response_time


