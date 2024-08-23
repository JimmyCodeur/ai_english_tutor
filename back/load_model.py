import ollama
import nltk
import httpx

def generate_ollama_response(model_name, prompt):
    # url = "http://ollama:11434"  # Utilisez le nom du service ici
    # print(f"Connecting to: {url} with model: {model_name} and prompt: {prompt}")
        
    response = ollama.generate(model=model_name, prompt=prompt)
    return response['response']


def generate_phi3_response(prompt):
    model_name = 'phi3'
    generated_response = generate_ollama_response(model_name, prompt)

    if not generated_response:
        return None

    if isinstance(generated_response, list):
        generated_response = ' '.join(generated_response) 

    sentences = nltk.tokenize.sent_tokenize(generated_response)    
    limited_response = ' '.join(sentences[:4])    
    return limited_response
