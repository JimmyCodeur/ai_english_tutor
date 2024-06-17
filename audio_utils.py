import sounddevice as sd
import wave
from datetime import datetime

def record_audio(duration, fs=16000, device=None):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"./user_audio/user_{current_time}.wav"
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16', device=device)
    sd.wait()
    save_audio(recording, file_path)
    return recording, file_path

def save_audio(audio, file_path, fs=16000):
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())
    print(f"Audio saved at {file_path}")
