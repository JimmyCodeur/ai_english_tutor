from load_model import generate_ollama_response
from text_to_speech import text_to_speech_audio
from log_conversation import log_conversation

model_name = 'phi3'
prompt = 'Tu es un professeur anglais et tu dois faire apprendre anglais juste en parlant avec une personne'

generated_response = generate_ollama_response(model_name, prompt)
print(generated_response)
text_to_speech_audio(generated_response)
log_conversation(prompt, generated_response)
