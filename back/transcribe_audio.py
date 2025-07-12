#fichier transcribe_audio.py
from faster_whisper import WhisperModel
import re
import time
import asyncio

async def transcribe_audio(file_path):
    start_time = time.time()
    model_size = "large-v2"
    asr_model = WhisperModel(model_size, device="cuda", compute_type="float16")
    
    segments, info = await asyncio.to_thread(asr_model.transcribe,
                                             file_path,
                                             language="en",
                                             beam_size=3,
                                             temperature=[0.0],
                                             patience=0.4,
                                             length_penalty=0.9,
                                             suppress_blank=False,
                                             word_timestamps=True,
                                             condition_on_previous_text=True)
    
    end_time = time.time()
    transcription_time = end_time - start_time
    
    transcribed_text = ' '.join([segment.text for segment in segments])
    transcribed_text_clean = re.sub(r'[^A-Za-z\s.,!?\'-]', '', transcribed_text)
    transcribed_text_clean = re.sub(r'\s+', ' ', transcribed_text_clean).strip()
    
    return transcribed_text_clean, transcription_time