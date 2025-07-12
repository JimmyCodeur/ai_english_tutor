#fichier back/tts_utils.py (VERSION MISE À JOUR)
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

# Import du nouveau système Edge-TTS
from back.edge_tts import edge_text_to_speech_with_fallback

# ========== CONFIGURATION MISE À JOUR ==========

# 🎯 MEILLEURS CHOIX RECOMMANDÉS (gardés pour compatibilité)
RECOMMENDED_VOICES = {
    # ⭐ NOUVELLES VOIX EDGE (prioritaires)
    "edge_mike": "edge-tts",
    "edge_alice": "edge-tts", 
    "edge_sarah": "edge-tts",
    
    # ⭐ ANCIENNES VOIX COQUI (fallback)
    "english_best": "tts_models/en/ljspeech/vits",
    "english_stable": "tts_models/en/ljspeech/tacotron2-DDC",
    "english_professional": "tts_models/en/sam/tacotron-DDC",
}

# 🎭 VOIX PAR PERSONNAGE (mis à jour avec Edge-TTS)
CHARACTER_VOICES = {
    "mike": "edge_mike",        # Edge-TTS voix masculine
    "alice": "edge_alice",      # Edge-TTS voix féminine amicale
    "sarah": "edge_sarah",      # Edge-TTS voix professionnelle
}

# 📋 FALLBACK HIERARCHY (gardé pour compatibilité Coqui)
FALLBACK_HIERARCHY = [
    "tts_models/en/ljspeech/tacotron2-DDC",
    "tts_models/en/ljspeech/vits",
    "tts_models/en/sam/tacotron-DDC",
    "tts_models/en/ljspeech/glow-tts",
]

# Ancien dictionnaire voices (gardé pour compatibilité)
voices = {
    # Nouvelles voix Edge
    "edge_mike": "edge-tts",
    "edge_alice": "edge-tts",
    "edge_sarah": "edge-tts",
    
    # Anciennes voix Coqui
    "english_ljspeech_tacotron2-DDC": "tts_models/en/ljspeech/tacotron2-DDC",
    "english_ljspeech_vits": "tts_models/en/ljspeech/vits",
    "english_sam_tacotron-DDC": "tts_models/en/sam/tacotron-DDC",
    "english_ljspeech_glow-tts": "tts_models/en/ljspeech/glow-tts",
    "english_vctk_vits": "tts_models/en/vctk/vits",
    "english_jenny_jenny": "tts_models/en/jenny/jenny",
    
    # Autres langues (gardées)
    "french_css10_vits": "tts_models/fr/css10/vits",
    "german_thorsten_vits": "tts_models/de/thorsten/vits",
    "spanish_css10_vits": "tts_models/es/css10/vits",
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
        print(f"📁 Répertoire créé/vérifié : {directory}")

def get_character_from_voice_key(voice_key):
    """Extraire le nom du personnage de la clé de voix"""
    if "mike" in voice_key.lower():
        return "mike"
    elif "alice" in voice_key.lower():
        return "alice"
    elif "sarah" in voice_key.lower():
        return "sarah"
    else:
        return "alice"  # Par défaut

async def text_to_speech_audio(generated_response, voice_key):
    """Version HYBRIDE: Edge-TTS prioritaire avec fallback Coqui"""
    try:
        print(f"🎙️ TTS Hybride: '{generated_response[:50]}...' avec {voice_key}")
        
        # 1. VÉRIFIER SI C'EST UNE VOIX EDGE
        if voice_key in ["edge_mike", "edge_alice", "edge_sarah"] or voice_key.startswith("edge_"):
            print(f"🔥 Utilisation Edge-TTS pour {voice_key}")
            
            # Extraire le personnage
            character = get_character_from_voice_key(voice_key)
            
            # Utiliser Edge-TTS avec fallback automatique
            return await edge_text_to_speech_with_fallback(generated_response, character)
        
        # 2. SI C'EST UNE VOIX CARACTÈRE, UTILISER EDGE
        if voice_key in CHARACTER_VOICES and CHARACTER_VOICES[voice_key].startswith("edge_"):
            print(f"🎭 Redirection vers Edge-TTS pour personnage {voice_key}")
            return await edge_text_to_speech_with_fallback(generated_response, voice_key)
        
        # 3. SINON, UTILISER L'ANCIEN SYSTÈME COQUI
        print(f"🤖 Utilisation Coqui TTS pour {voice_key}")
        return await coqui_text_to_speech_audio(generated_response, voice_key)
        
    except Exception as e:
        print(f"❌ Erreur TTS Hybride: {e}")
        print("🆘 Tentative Edge-TTS d'urgence...")
        
        try:
            # En cas d'erreur, toujours essayer Edge-TTS
            character = get_character_from_voice_key(voice_key)
            return await edge_text_to_speech_with_fallback(generated_response, character)
        except Exception as emergency_error:
            print(f"💀 Edge-TTS d'urgence échoué: {emergency_error}")
            raise e

async def coqui_text_to_speech_audio(generated_response, voice_key):
    """ANCIENNE FONCTION TTS COQUI (renommée pour éviter les conflits)"""
    try:
        print(f"🤖 Coqui TTS: '{generated_response}' avec {voice_key}")
        
        if voice_key not in voices:
            print(f"⚠️ Clé de voix invalide : {voice_key}, utilisation du fallback")
            voice_key = "english_ljspeech_tacotron2-DDC"
        
        # S'assurer que le répertoire existe
        audio_dir = "./audio/teacher"
        Path(audio_dir).mkdir(parents=True, exist_ok=True)
        
        model_name = voices[voice_key]
        print(f"🤖 Chargement du modèle Coqui : {model_name}")
        
        # Initialiser TTS Coqui
        tts = TTS(model_name=model_name, progress_bar=False, gpu=True)
        
        # Nettoyer le texte
        if isinstance(generated_response, tuple):
            generated_response = generated_response[0]
        
        if not isinstance(generated_response, str):
            generated_response = str(generated_response)
        
        generated_response = generated_response.strip()
        if not generated_response:
            raise ValueError("Le texte à synthétiser est vide")
        
        # Générer l'audio
        start_time = time.time()
        wav_data = await asyncio.to_thread(tts.tts, generated_response)
        generation_time = time.time() - start_time
        
        # Traitement audio
        if isinstance(wav_data, tuple):
            wav_data = wav_data[0]
        
        if wav_data is None or len(wav_data) == 0:
            raise ValueError("Les données audio générées sont vides")
        
        # Normalisation
        wav_data_np = np.array(wav_data, dtype=np.float32)
        max_val = np.max(np.abs(wav_data_np))
        if max_val > 0:
            wav_data_np = wav_data_np / max_val
        
        # Filtrage
        try:
            cutoff_freq = 8000
            wav_data_filtered = lowpass_filter(wav_data_np, cutoff_freq, 22050)
            wav_data_pcm = np.int16(wav_data_filtered * 32767)
        except Exception:
            wav_data_pcm = np.int16(wav_data_np * 32767)
        
        # Sauvegarder
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"coqui_{current_time}.wav"
        audio_file_path = os.path.join(audio_dir, filename)
        
        with wave.open(audio_file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(wav_data_pcm.tobytes())
        
        # Calculer durée
        try:
            audio = AudioSegment.from_file(audio_file_path)
            duration = len(audio) / 1000.0
        except Exception:
            duration = len(wav_data_pcm) / 22050.0
        
        print(f"✅ Coqui TTS terminé en {generation_time:.2f}s, durée: {duration:.2f}s")
        return audio_file_path, duration
        
    except Exception as e:
        print(f"❌ Erreur Coqui TTS: {e}")
        raise

def ensure_tts_model_downloaded(model_name):
    """Fonction gardée pour compatibilité avec Coqui"""
    try:
        print(f"🔄 Vérification du modèle Coqui: {model_name}")
        tts = TTS(model_name=model_name, progress_bar=True, gpu=True)
        print(f"✅ Modèle Coqui {model_name} prêt")
        return tts
    except Exception as e:
        print(f"❌ Erreur avec le modèle {model_name}: {e}")
        
        # Fallback
        fallback_model = "tts_models/en/ljspeech/tacotron2-DDC"
        try:
            tts = TTS(model_name=fallback_model, progress_bar=True, gpu=True)
            print(f"✅ Modèle fallback {fallback_model} prêt")
            return tts
        except Exception as fallback_error:
            print(f"❌ Erreur même avec fallback: {fallback_error}")
            raise

# Fonctions utilitaires
def get_best_voice_for_character(character_name):
    """Obtenir la meilleure voix (Edge) pour un personnage"""
    edge_mapping = {
        "mike": "edge_mike",
        "alice": "edge_alice", 
        "sarah": "edge_sarah"
    }
    return edge_mapping.get(character_name, "edge_alice")

def is_edge_voice(voice_key):
    """Vérifier si une voix utilise Edge-TTS"""
    return voice_key.startswith("edge_") or voice_key in ["edge_mike", "edge_alice", "edge_sarah"]