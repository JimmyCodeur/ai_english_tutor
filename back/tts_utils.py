from TTS.api import TTS
import numpy as np
import wave
from datetime import datetime
from filters_audio import lowpass_filter

# tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=True)
# tts_model.to("cuda")

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


def text_to_speech_audio(generated_response, voice_key):
    if voice_key not in voices:
        raise ValueError(f"Invalid voice key: {voice_key}")

    model_name = voices[voice_key]

    # Initialisation du modèle TTS sans spécification du locuteur
    tts = TTS(model_name=model_name, progress_bar=False, gpu=True)

    # Synthèse vocale à partir du texte généré
    wav_data = tts.tts(generated_response)

    # Traitement audio (normalisation, filtrage passe-bas, etc.)
    wav_data_np = np.array(wav_data, dtype=np.float32)
    wav_data_np = wav_data_np / np.max(np.abs(wav_data_np))

    cutoff_freq = 8000  # Fréquence de coupure du filtre passe-bas
    wav_data_filtered = lowpass_filter(wav_data_np, cutoff_freq, 22050)  # Appliquez votre filtre passe-bas ici

    wav_data_pcm = np.int16(wav_data_filtered * 32767)

    # Création du chemin et écriture du fichier audio WAV
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_file_path = f"./audio/teacher/teacher_{current_time}.wav"

    with wave.open(audio_file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(wav_data_pcm.tobytes())

    return audio_file_path