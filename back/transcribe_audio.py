from faster_whisper import WhisperModel
import re

def transcribe_audio(file_path):
    model_size = "large-v3"
    asr_model = WhisperModel(model_size, device="cuda", compute_type="float16")
    segments, info = asr_model.transcribe(file_path, beam_size=5,language="en")
    
    transcribed_text = ' '.join([segment.text for segment in segments])
    
    transcribed_text_clean = re.sub(r'[^A-Za-z\s.,!?\'-]', '', transcribed_text)
    transcribed_text_clean = re.sub(r'\s+', ' ', transcribed_text_clean).strip()

    return transcribed_text_clean