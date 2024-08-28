import wave
from datetime import datetime
from fastapi import UploadFile
import base64
import os
import subprocess
import soundfile as sf
import noisereduce
from scipy.signal import butter, lfilter

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
        f.write(audio_file.read())
    return audio_path

def file_to_base64(file_path):
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode("utf-8")

def delete_audio_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_path} deleted successfully.")
        else:
            print(f"File {file_path} does not exist.")
    except OSError as e:
        print(f"Error: {e.strerror}. Could not delete the file: {file_path}")


def process_audio_file(audio_path, filename):
    converted_audio_path = f"./audio/user/converted_{filename}"
    if os.path.exists(converted_audio_path):
        os.remove(converted_audio_path)
    subprocess.run(['ffmpeg', '-i', audio_path, '-acodec', 'pcm_s16le', '-ar', '44100', converted_audio_path])

    audio_data, sr = sf.read(converted_audio_path)
    reduced_noise = noisereduce.reduce_noise(audio_data, sr)

    denoised_audio_path = f"./audio/user/denoised_converted_{filename}"
    sf.write(denoised_audio_path, reduced_noise, sr)

    return denoised_audio_path

def is_valid_audio_file(file: UploadFile) -> bool:
    valid_extensions = ['wav', 'mp3', 'flac', 'ogg']
    return file.filename.split('.')[-1].lower() in valid_extensions

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

