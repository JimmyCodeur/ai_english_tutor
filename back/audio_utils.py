import sounddevice as sd
import wave
from datetime import datetime
from fastapi import UploadFile
import base64

# def record_audio(duration, fs=16000, device=None):
#     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#     file_path = f"./user_audio/user_{current_time}.wav"
#     print("Recording...")
#     recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16', device=device)
#     sd.wait()
#     save_audio(recording, file_path)
#     return recording, file_path

def save_audio(audio, file_path, fs=16000):
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())
    print(f"Audio saved at {file_path}")

def save_user_audio(audio_file: UploadFile):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_path = f"./audio/user/user_{current_time}.wav"
    with open(audio_path, "wb") as f:
        f.write(audio_file.file.read())
    return audio_path

def file_to_base64(file_path):
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode("utf-8")
