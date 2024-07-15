import ollama
import nltk

def generate_ollama_response(model_name, prompt):
    response = ollama.generate(model=model_name, prompt=prompt)
    return response['response']

def generate_phi3_response(prompt):
    model_name = 'phi3'
    generated_response = generate_ollama_response(model_name, prompt)

    if isinstance(generated_response, list):
        generated_response = ' '.join(generated_response) 

    sentences = nltk.tokenize.sent_tokenize(generated_response)    
    limited_response = ' '.join(sentences[:4])    
    return limited_response

def generate_phi3_response_with_history(prompt, user_history):
    # Create a structured prompt that includes the full conversation history
    conversation_context = " ".join(user_history)  # Converts history into a single string
    full_prompt = f"{prompt}\n\n{conversation_context}"

    model_name = 'phi3'
    generated_response = generate_ollama_response(model_name, full_prompt)

    if isinstance(generated_response, list):
        generated_response = ' '.join(generated_response)

    # Use nltk to break down the response into sentences
    sentences = nltk.tokenize.sent_tokenize(generated_response)    
    limited_response = ' '.join(sentences[:4])  # Limit the response to the first few sentences to keep it concise


def generate_ai_response_alice(previous_input):
    # Create a structured prompt that includes the full conversation history
    conversation_context = " ".join(user_history)  # Converts history into a single string
    full_prompt = f"{ai_prompt}\n\n{conversation_context}"

    model_name = 'phi3'
    generated_response = generate_ollama_response(model_name, full_prompt)

    if isinstance(generated_response, list):
        generated_response = ' '.join(generated_response)

    # Use nltk to break down the response into sentences
    sentences = nltk.tokenize.sent_tokenize(generated_response)    
    limited_response = ' '.join(sentences[:4])  # Limit the response to the first few sentences to keep it concise

    return limited_response