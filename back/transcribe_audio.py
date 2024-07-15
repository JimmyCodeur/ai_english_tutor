from faster_whisper import WhisperModel
import re

def transcribe_audio(file_path):
    model_size = "large-v3"
    asr_model = WhisperModel(model_size, device="cuda", compute_type="float16")
    segments, info = asr_model.transcribe(        
        file_path, 
        language="en", 
        beam_size=10,               # Increase beam size for better accuracy
        temperature=[0.0],          # Set temperature to a single low value
        patience=1,                 # Set patience factor to 1 for strict beam search
        length_penalty=1,           # No length penalty
        repetition_penalty=1,       # No repetition penalty
        no_repeat_ngram_size=0,     # Allow repetitions
        suppress_blank=False,       # Do not suppress blank tokens
        word_timestamps=True,       # Enable word timestamps for detailed output
        condition_on_previous_text=True,
        )
    
    transcribed_text = ' '.join([segment.text for segment in segments])
    
    transcribed_text_clean = re.sub(r'[^A-Za-z\s.,!?\'-]', '', transcribed_text)
    transcribed_text_clean = re.sub(r'\s+', ' ', transcribed_text_clean).strip()

    return transcribed_text_clean