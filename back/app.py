from system_utils import check_cuda

from tts_utils import text_to_speech_audio
from load_model import generate_ollama_response
from log_conversation import log_conversation
from faster_whisper import WhisperModel
from TTS.api import TTS
import os
import re

model_name = 'phi3'
model_size = "large-v2"  
asr_model = WhisperModel(model_size, device="cuda", compute_type="float16")

tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=True)
tts_model.to("cuda")

def transcribe_audio(file_path):
    model_size = "large-v3"
    asr_model = WhisperModel(model_size, device="cuda", compute_type="float16")
    segments, info = asr_model.transcribe(file_path, beam_size=5)
    
    # Combine segments into a single string
    transcribed_text = ' '.join([segment.text for segment in segments])
    
    # Replace unwanted characters and clean the text
    transcribed_text_clean = re.sub(r'[^A-Za-z0-9\s.,!?\'-]', '', transcribed_text)  # Keep only alphanumeric characters and common punctuation
    transcribed_text_clean = re.sub(r'\s+', ' ', transcribed_text_clean).strip()  # Remove extra whitespace and strip leading/trailing spaces

    return transcribed_text_clean

def main():
    check_cuda()
    audio_device_index = 1

    while True:
        audio, audio_path = record_audio(duration=10, device=audio_device_index)
        
        user_input = transcribe_audio(audio_path).strip()
        
        if not user_input:
            print("L'utilisateur n'a rien dit.")
            continue
        
        print(f"User input: {user_input}")

        if "stop" in user_input.lower():
            print("Conversation arrêtée.")
            break
        
        
        prompt = (
            "As an English-speaking correspondent, your mission is to assist a French-speaking user in learning and enhancing their English skills through engaging and adaptive conversations.\n"
            "1. Engage the user in natural, immersive English conversations to foster language practice and fluency.\n"
            "2. Assess the user's English proficiency level and tailor your conversation accordingly, balancing challenge and support.\n"
            "3. Provide gentle corrections and explanations in English when needed, helping the user learn from their mistakes.\n"
            "4. Track the user's progress over time and adapt your teaching approach to maximize learning efficiency.\n"
            "5. Correct any English questions from the user if they are incorrect. Encourage them to rephrase or clarify their question for better understanding.\n"
            "6. If the user responds in French, translate their response to English and prompt them to repeat it in English for continuous practice.\n\n"
            "Begin by introducing a topic of interest or asking about the user's day. Encourage them to express themselves in English as much as possible.\n"
            "If the user makes an error, kindly correct them and explain the correction in English. Gradually introduce new vocabulary and more complex sentence structures as their confidence grows.\n\n"
            "If the user seems to struggle, simplify your language and offer supportive feedback. If they express frustration or confusion, briefly switch to French to clarify and then return to English promptly.\n\n"
            "Remember, the goal is to create a supportive and encouraging environment where the user can practice and improve their English skills effectively.\n\n"
            f"User: {user_input}\n "
        )
        
        generated_response = generate_ollama_response(model_name, prompt)
        
        if isinstance(generated_response, list):
            generated_response = ' '.join(generated_response)
        
        print(generated_response)
        
        audio_file_path = text_to_speech_audio(generated_response)
        print(audio_file_path)

        log_conversation(prompt, generated_response)
    
        os.system(f"aplay {audio_file_path}")

        continue_conversation = input("Voulez-vous continuer la conversation? (oui/non): ").strip().lower()
        if continue_conversation not in ["oui", "yes"]:
            print("Conversation arrêtée.")
            break
        
if __name__ == "__main__":
    main()

