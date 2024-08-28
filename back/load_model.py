import ollama
import nltk
import httpx
from fastapi import HTTPException
import json

def generate_ollama_response(model_name, prompt):
    url = "http://ollama:11434/api/generate"  # Assurez-vous que c'est le bon endpoint
    try:
        response = httpx.post(url, json={"model": model_name, "prompt": prompt}, timeout=None)
        response.raise_for_status()  # Vérifie que la réponse est 200 OK
        
        # Lire la réponse par morceaux et les combiner
        json_lines = response.text.strip().splitlines()
        combined_response = ""

        for line in json_lines:
            # Vérifier si chaque ligne est bien formatée en JSON
            try:
                json_line = json.loads(line)
                if 'response' in json_line:
                    combined_response += json_line['response']
            except json.JSONDecodeError as e:
                print(f"Erreur lors de la tentative de décodage JSON: {e}")
                continue  # Ignorer les lignes non valides

        print("Réponse combinée:", combined_response)
        return combined_response.strip()

    except httpx.ConnectError as exc:
        print("Erreur de connexion à Ollama API:", exc)
        raise HTTPException(status_code=500, detail="Impossible de se connecter à Ollama API")
    except httpx.HTTPStatusError as exc:
        print(f"Erreur HTTP lors de la connexion à Ollama API: {exc}")
        raise HTTPException(status_code=exc.response.status_code, detail=f"Erreur avec Ollama API: {exc}")



def generate_phi3_response(prompt):
    model_name = 'phi3'
    generated_response = generate_ollama_response(model_name, prompt)

    if not generated_response:
        return None

    if isinstance(generated_response, tuple):
        # Si la réponse est un tuple, convertissez-la en chaîne
        generated_response = ' '.join(generated_response)

    sentences = nltk.tokenize.sent_tokenize(generated_response)    
    limited_response = ' '.join(sentences[:4])    
    return limited_response
