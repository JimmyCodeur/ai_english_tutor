from faster_whisper import WhisperModel
import re
import time
from metrics import log_metrics_transcription_time

def transcribe_audio(file_path):
    model_size = "large-v3"
    asr_model = WhisperModel(model_size, device="cuda", compute_type="float16")
    start_time = time.time()
    segments, info = asr_model.transcribe(
        file_path,
        language="en",
        beam_size=10,
        temperature=[0.0],
        patience=1,
        length_penalty=1,
        repetition_penalty=1,
        no_repeat_ngram_size=0,
        suppress_blank=False,
        word_timestamps=True,
        condition_on_previous_text=True,
    )
    end_time = time.time()
    transcribed_text = ' '.join([segment.text for segment in segments])
    transcribed_text_clean = re.sub(r'[^A-Za-z\s.,!?\'-]', '', transcribed_text)
    transcribed_text_clean = re.sub(r'\s+', ' ', transcribed_text_clean).strip()

    transcription_time = log_metrics_transcription_time(start_time, end_time)

    return transcribed_text_clean, transcription_time