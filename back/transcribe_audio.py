from faster_whisper import WhisperModel
import re
import time
from back.metrics import log_metrics_transcription_time
import asyncio

async def transcribe_audio(file_path):
    start_time = time.time()
    model_size = "large-v2"  # Example smaller model for faster performance
    asr_model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # Remove the repetition_penalty argument
    segments, info = await asyncio.to_thread(asr_model.transcribe,
                                             file_path,
                                             language="en",
                                             beam_size=3,  # Reduced beam size for faster transcription
                                             temperature=[0.0],
                                             patience=0.4,
                                             length_penalty=0.9,
                                             suppress_blank=False,
                                             word_timestamps=True,
                                             condition_on_previous_text=True)

    end_time = time.time()
    transcribed_text = ' '.join([segment.text for segment in segments])
    transcribed_text_clean = re.sub(r'[^A-Za-z\s.,!?\'-]', '', transcribed_text)
    transcribed_text_clean = re.sub(r'\s+', ' ', transcribed_text_clean).strip()

    transcription_time = log_metrics_transcription_time(start_time, end_time)
    return transcribed_text_clean, transcription_time
