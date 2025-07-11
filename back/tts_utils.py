from TTS.api import TTS
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from back.audio_utils import lowpass_filter
from pydub import AudioSegment
import time
import asyncio
import wave
from back.metrics import log_custom_metric

voices = {
    "xtts_v2": "tts_models/multilingual/multi-dataset/xtts_v2",
    "xtts_v1.1": "tts_models/multilingual/multi-dataset/xtts_v1.1",
    "your_tts": "tts_models/multilingual/multi-dataset/your_tts",
    "bark": "tts_models/multilingual/multi-dataset/bark",
    "bulgarian": "tts_models/bg/cv/vits",
    "czech": "tts_models/cs/cv/vits",
    "danish": "tts_models/da/cv/vits",
    "estonian": "tts_models/et/cv/vits",
    "irish": "tts_models/ga/cv/vits",
    "english_ek1": "tts_models/en/ek1/tacotron2",
    "english_ljspeech_tacotron2-DDC": "tts_models/en/ljspeech/tacotron2-DDC",
    "english_ljspeech_tacotron2-DDC_ph": "tts_models/en/ljspeech/tacotron2-DDC_ph",
    "english_ljspeech_glow-tts": "tts_models/en/ljspeech/glow-tts",
    "english_ljspeech_speedy-speech": "tts_models/en/ljspeech/speedy-speech",
    "english_ljspeech_tacotron2-DCA": "tts_models/en/ljspeech/tacotron2-DCA",
    "english_ljspeech_vits": "tts_models/en/ljspeech/vits",
    "english_ljspeech_vits_neon": "tts_models/en/ljspeech/vits--neon",
    "english_ljspeech_fast_pitch": "tts_models/en/ljspeech/fast_pitch",
    "english_ljspeech_overflow": "tts_models/en/ljspeech/overflow",
    "english_ljspeech_neural_hmm": "tts_models/en/ljspeech/neural_hmm",
    "english_vctk_vits": "tts_models/en/vctk/vits",
    "english_vctk_fast_pitch": "tts_models/en/vctk/fast_pitch",
    "english_sam_tacotron-DDC": "tts_models/en/sam/tacotron-DDC",
    "english_blizzard2013_capacitron-t2-c50": "tts_models/en/blizzard2013/capacitron-t2-c50",
    "english_blizzard2013_capacitron-t2-c150_v2": "tts_models/en/blizzard2013/capacitron-t2-c150_v2",
    "english_multi-dataset_tortoise-v2": "tts_models/en/multi-dataset/tortoise-v2",
    "english_jenny_jenny": "tts_models/en/jenny/jenny",
    "spanish_mai_tacotron2-DDC": "tts_models/es/mai/tacotron2-DDC",
    "spanish_css10_vits": "tts_models/es/css10/vits",
    "french_mai_tacotron2-DDC": "tts_models/fr/mai/tacotron2-DDC",
    "french_css10_vits": "tts_models/fr/css10/vits",
    "ukrainian_glow-tts": "tts_models/uk/mai/glow-tts",
    "ukrainian_vits": "tts_models/uk/mai/vits",
    "chinese_baker_tacotron2-DDC-GST": "tts_models/zh-CN/baker/tacotron2-DDC-GST",
    "dutch_mai_tacotron2-DDC": "tts_models/nl/mai/tacotron2-DDC",
    "dutch_css10_vits": "tts_models/nl/css10/vits",
    "german_thorsten_tacotron2-DCA": "tts_models/de/thorsten/tacotron2-DCA",
    "german_thorsten_vits": "tts_models/de/thorsten/vits",
    "german_thorsten_tacotron2-DDC": "tts_models/de/thorsten/tacotron2-DDC",
    "german_css10_vits_neon": "tts_models/de/css10/vits-neon",
    "japanese_kokoro_tacotron2-DDC": "tts_models/ja/kokoro/tacotron2-DDC",
    "turkish_common-voice_glow-tts": "tts_models/tr/common-voice/glow-tts",
    "italian_female_glow-tts": "tts_models/it/mai_female/glow-tts",
    "italian_female_vits": "tts_models/it/mai_female/vits",
    "italian_male_glow-tts": "tts_models/it/mai_male/glow-tts",
    "italian_male_vits": "tts_models/it/mai_male/vits",
    "ewe_openbible_vits": "tts_models/ewe/openbible/vits",
    "hau_openbible_vits": "tts_models/hau/openbible/vits",
    "lin_openbible_vits": "tts_models/lin/openbible/vits",
    "tw_akuapem_openbible_vits": "tts_models/tw_akuapem/openbible/vits",
    "tw_asante_openbible_vits": "tts_models/tw_asante/openbible/vits",
    "yoruba_openbible_vits": "tts_models/yor/openbible/vits",
    "hungarian_css10_vits": "tts_models/hu/css10/vits",
    "greek_cv_vits": "tts_models/el/cv/vits",
    "finnish_css10_vits": "tts_models/fi/css10/vits",
    "croatian_cv_vits": "tts_models/hr/cv/vits",
    "lithuanian_cv_vits": "tts_models/lt/cv/vits",
    "latvian_cv_vits": "tts_models/lv/cv/vits",
    "maltese_cv_vits": "tts_models/mt/cv/vits",
    "polish_female_vits": "tts_models/pl/mai_female/vits",
    "portuguese_cv_vits": "tts_models/pt/cv/vits",
    "romanian_cv_vits": "tts_models/ro/cv/vits",
    "slovak_cv_vits": "tts_models/sk/cv/vits",
    "slovenian_cv_vits": "tts_models/sl/cv/vits",
    "swedish_cv_vits": "tts_models/sv/cv/vits",
    "catalan_custom_vits": "tts_models/ca/custom/vits",
    "persian_custom_glow-tts": "tts_models/fa/custom/glow-tts",
    "bengali_male_custom_vits": "tts_models/bn/custom/vits-male",
    "bengali_female_custom_vits": "tts_models/bn/custom/vits-female",
    "belarusian_common-voice_glow-tts": "tts_models/be/common-voice/glow-tts"
}

vocoder_models = {
    "libri-tts_wavegrad": "vocoder_models/universal/libri-tts/wavegrad",
    "libri-tts_fullband-melgan": "vocoder_models/universal/libri-tts/fullband-melgan",
    "ek1_wavegrad": "vocoder_models/en/ek1/wavegrad",
    "ljspeech_multiband-melgan": "vocoder_models/en/ljspeech/multiband-melgan",
    "ljspeech_hifigan_v2": "vocoder_models/en/ljspeech/hifigan_v2",
    "ljspeech_univnet": "vocoder_models/en/ljspeech/univnet",
    "blizzard2013_hifigan_v2": "vocoder_models/en/blizzard2013/hifigan_v2",
    "vctk_hifigan_v2": "vocoder_models/en/vctk/hifigan_v2",
    "sam_hifigan_v2": "vocoder_models/en/sam/hifigan_v2",
    "mai_parallel-wavegan": "vocoder_models/nl/mai/parallel-wavegan",
    "thorsten_wavegrad": "vocoder_models/de/thorsten/wavegrad",
    "thorsten_fullband-melgan": "vocoder_models/de/thorsten/fullband-melgan",
    "thorsten_hifigan_v1": "vocoder_models/de/thorsten/hifigan_v1",
    "kokoro_hifigan_v1": "vocoder_models/ja/kokoro/hifigan_v1",
    "uk_mai_multiband-melgan": "vocoder_models/uk/mai/multiband-melgan",
    "tr_common-voice_hifigan": "vocoder_models/tr/common-voice/hifigan",
    "be_common-voice_hifigan": "vocoder_models/be/common-voice/hifigan"
}

def ensure_audio_directories():
    """Créer les répertoires audio s'ils n'existent pas"""
    directories = [
        './audio',
        './audio/teacher',
        './audio/user',
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Répertoire créé/vérifié : {directory}")

async def text_to_speech_audio(generated_response, voice_key):
    try:
        print(f"Début TTS pour : '{generated_response}' avec la voix : {voice_key}")
        
        if voice_key not in voices:
            raise ValueError(f"Clé de voix invalide : {voice_key}")
        
        # S'assurer que le répertoire existe
        audio_dir = "./audio/teacher"
        Path(audio_dir).mkdir(parents=True, exist_ok=True)
        print(f"Répertoire audio vérifié : {audio_dir}")
        
        model_name = voices[voice_key]
        print(f"Chargement du modèle TTS : {model_name}")
        
        # Initialiser TTS
        tts = TTS(model_name=model_name, progress_bar=False, gpu=True)
        
        # Nettoyer le texte d'entrée
        if isinstance(generated_response, tuple):
            print(f"Le texte est un tuple: {generated_response}, extraction du premier élément")
            generated_response = generated_response[0]
        
        if not isinstance(generated_response, str):
            generated_response = str(generated_response)
        
        generated_response = generated_response.strip()
        if not generated_response:
            raise ValueError("Le texte à synthétiser est vide")
        
        print(f"Texte nettoyé : '{generated_response}'")
        
        start_time_total = time.time()
        start_time_tts = time.time()
        
        # Générer l'audio dans un thread séparé
        print("Génération de l'audio TTS...")
        wav_data = await asyncio.to_thread(tts.tts, generated_response)
        tts_time = time.time() - start_time_tts
        await log_custom_metric("Generate TTS audio time", tts_time)
        
        print(f"Audio TTS généré en {tts_time:.2f}s")
        
        # Traitement des données audio
        if isinstance(wav_data, tuple):
            print(f"wav_data est un tuple, contenu : {wav_data}")
            wav_data = wav_data[0]
        
        if wav_data is None or len(wav_data) == 0:
            raise ValueError("Les données audio générées sont vides")
        
        # Normalisation et filtrage
        wav_data_np = np.array(wav_data, dtype=np.float32)
        
        # Éviter la division par zéro
        max_val = np.max(np.abs(wav_data_np))
        if max_val > 0:
            wav_data_np = wav_data_np / max_val
        
        cutoff_freq = 8000
        wav_data_filtered = lowpass_filter(wav_data_np, cutoff_freq, 22050)
        wav_data_pcm = np.int16(wav_data_filtered * 32767)
        
        # Créer le nom de fichier unique
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Microseconds pour unicité
        filename = f"teacher_{current_time}.wav"
        audio_file_path = os.path.join(audio_dir, filename)
        
        print(f"Écriture du fichier audio : {audio_file_path}")
        
        # Écrire le fichier WAV
        try:
            with wave.open(audio_file_path, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(22050)  # Sample rate
                wf.writeframes(wav_data_pcm.tobytes())
        except Exception as wav_error:
            raise ValueError(f"Erreur lors de l'écriture du fichier WAV : {wav_error}")
        
        # Vérifier que le fichier a été créé
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Le fichier audio n'a pas été créé : {audio_file_path}")
        
        # Vérifier la taille du fichier
        file_size = os.path.getsize(audio_file_path)
        if file_size == 0:
            raise ValueError(f"Le fichier audio créé est vide : {audio_file_path}")
        
        print(f"Fichier audio créé avec succès : {audio_file_path} ({file_size} bytes)")
        
        # Calculer la durée
        try:
            audio = AudioSegment.from_file(audio_file_path)
            duration = len(audio) / 1000.0  # Durée en secondes
        except Exception as duration_error:
            print(f"Erreur lors du calcul de la durée : {duration_error}")
            # Estimation approximative basée sur le taux d'échantillonnage
            duration = len(wav_data_pcm) / 22050.0
        
        total_time = time.time() - start_time_total
        print(f"TTS terminé en {total_time:.2f}s, durée audio : {duration:.2f}s")
        
        return audio_file_path, duration
        
    except Exception as e:
        print(f"Erreur dans text_to_speech_audio : {e}")
        import traceback
        print(traceback.format_exc())
        raise