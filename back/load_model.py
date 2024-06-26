import ollama

def generate_ollama_response(model_name, prompt):
    """
    Génère une réponse à partir d'un modèle Ollama et d'une phrase donnée.
    
    Args:
        model_name (str): Le nom du modèle Ollama à utiliser.
        prompt (str): La phrase d'entrée pour générer la réponse.
    
    Returns:
        str: La réponse générée par le modèle.
    """
    response = ollama.generate(model=model_name, prompt=prompt)
    return response['response']
