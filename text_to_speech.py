from gtts import gTTS
import os
from datetime import datetime

def text_to_speech_audio(generated_response):
    tts = gTTS(text=generated_response, lang='fr')
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S") 
    audio_file_path = f"./teacher/teacher_{current_time}.mp3" 
    tts.save(audio_file_path)

