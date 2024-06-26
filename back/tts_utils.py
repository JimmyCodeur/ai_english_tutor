from TTS.api import TTS
import numpy as np
import wave
from datetime import datetime
from filters_audio import lowpass_filter

tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=True)
tts_model.to("cuda")

def text_to_speech_audio(generated_response):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_file_path = f"./back/ai_audio/teacher_{current_time}.wav"
    wav_data = tts_model.tts(generated_response)
    wav_data_np = np.array(wav_data, dtype=np.float32)
    wav_data_np = wav_data_np / np.max(np.abs(wav_data_np))
    cutoff_freq = 8000 
    wav_data_filtered = lowpass_filter(wav_data_np, cutoff_freq, 22050)
    wav_data_pcm = np.int16(wav_data_filtered * 32767)
    fade_duration = 0.1 
    fade_samples = int(fade_duration * 22050)
    wav_data_pcm[:fade_samples] = np.linspace(0, wav_data_pcm[fade_samples], fade_samples)
    wav_data_pcm[-fade_samples:] = np.linspace(wav_data_pcm[-fade_samples], 0, fade_samples)
    with wave.open(audio_file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050) 
        wf.writeframes(wav_data_pcm.tobytes())
    
    return audio_file_path
    
