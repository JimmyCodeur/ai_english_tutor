#fichier back/edge_tts.py
import edge_tts
import asyncio
import time
from pathlib import Path
import os
from pydub import AudioSegment

# Voix Edge par personnage (toutes gratuites et de haute qualité)
EDGE_VOICES = {
    "mike": {
        "voice": "en-US-BrianNeural",           # Voix masculine naturelle
        "rate": "+0%",                          # Vitesse normale
        "volume": "+0%",                        # Volume normal
        "pitch": "+0Hz"                         # Tonalité normale
    },
    "alice": {
        "voice": "en-US-JennyNeural",           # Voix féminine amicale
        "rate": "+5%",                          # Légèrement plus rapide
        "volume": "+10%",                       # Un peu plus fort
        "pitch": "+20Hz"                        # Tonalité plus aiguë
    },
    "sarah": {
        "voice": "en-US-AriaNeural",            # Voix professionnelle
        "rate": "+0%",                          # Vitesse normale
        "volume": "+0%",                        # Volume normal
        "pitch": "+0Hz"                         # Tonalité normale
    }
}

# Voix alternatives par caractère (si les principales ne fonctionnent pas)
EDGE_FALLBACK_VOICES = {
    "mike": ["en-US-DavisNeural", "en-US-BrandonNeural", "en-US-ChristopherNeural"],
    "alice": ["en-US-MonicaNeural", "en-US-AmberNeural", "en-US-AvaNeural"],
    "sarah": ["en-US-SaraNeural", "en-US-ElizabethNeural", "en-US-MichelleNeural"]
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

async def get_available_edge_voices():
    """Lister toutes les voix Edge disponibles pour debug"""
    try:
        voices = await edge_tts.list_voices()
        english_voices = [v for v in voices if v['Locale'].startswith('en-')]
        print(f"🎭 {len(english_voices)} voix anglaises disponibles")
        return english_voices
    except Exception as e:
        print(f"❌ Erreur listage voix: {e}")
        return []

async def test_voice_availability(voice_name):
    """Tester si une voix spécifique est disponible"""
    try:
        test_text = "Test"
        communicate = edge_tts.Communicate(text=test_text, voice=voice_name)
        # Juste tester la création sans sauvegarder
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                return True
        return False
    except Exception:
        return False

async def edge_text_to_speech(text, character="alice"):
    """Génération audio gratuite avec Edge-TTS - Version robuste"""
    try:
        print(f"🎤 Edge-TTS: Début génération pour {character}")
        print(f"📝 Texte: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        # S'assurer que les répertoires existent
        ensure_audio_directories()
        
        # Nettoyer et valider le texte
        if not text or not text.strip():
            raise ValueError("Texte vide fourni")
        
        text = text.strip()
        
        # Limiter la longueur pour éviter les erreurs
        if len(text) > 1000:
            text = text[:1000] + "..."
            print("⚠️ Texte tronqué à 1000 caractères")
        
        # Obtenir la configuration de la voix
        voice_config = EDGE_VOICES.get(character, EDGE_VOICES["alice"])
        primary_voice = voice_config["voice"]
        
        print(f"🎭 Voix sélectionnée: {primary_voice}")
        
        # Tenter avec la voix principale
        selected_voice = primary_voice
        
        # Si la voix principale échoue, essayer les alternatives
        if not await test_voice_availability(primary_voice):
            print(f"⚠️ Voix principale {primary_voice} non disponible, test alternatives...")
            
            fallback_voices = EDGE_FALLBACK_VOICES.get(character, [])
            for fallback_voice in fallback_voices:
                if await test_voice_availability(fallback_voice):
                    selected_voice = fallback_voice
                    print(f"✅ Voix alternative trouvée: {selected_voice}")
                    break
            else:
                # Utiliser une voix par défaut si aucune alternative ne fonctionne
                selected_voice = "en-US-AriaNeural"  # Voix très fiable
                print(f"🔄 Utilisation voix par défaut: {selected_voice}")
        
        # Créer le communicateur Edge-TTS avec les paramètres
        communicate = edge_tts.Communicate(
            text=text,
            voice=selected_voice,
            rate=voice_config["rate"],
            volume=voice_config["volume"],
            pitch=voice_config["pitch"]
        )
        
        # Générer le nom de fichier unique
        timestamp = int(time.time() * 1000)  # Millisecondes pour plus d'unicité
        filename = f"edge_{character}_{timestamp}.wav"
        audio_path = f"./audio/teacher/{filename}"
        
        print(f"💾 Sauvegarde vers: {audio_path}")
        
        # Sauvegarder l'audio
        start_time = time.time()
        await communicate.save(audio_path)
        generation_time = time.time() - start_time
        
        # Vérifier que le fichier a été créé
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Le fichier audio n'a pas été créé: {audio_path}")
        
        # Vérifier la taille du fichier
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            raise ValueError(f"Le fichier audio créé est vide: {audio_path}")
        
        print(f"✅ Fichier créé: {audio_path} ({file_size} bytes)")
        
        # Calculer la durée avec pydub
        try:
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000.0  # Durée en secondes
        except Exception as duration_error:
            print(f"⚠️ Erreur calcul durée: {duration_error}")
            # Estimation approximative: ~150 mots par minute
            word_count = len(text.split())
            duration = (word_count / 150) * 60
        
        print(f"✅ Edge-TTS terminé en {generation_time:.2f}s, durée audio: {duration:.2f}s")
        
        return audio_path, duration
        
    except Exception as e:
        print(f"❌ Erreur Edge-TTS: {e}")
        import traceback
        print("🔍 Traceback Edge-TTS:")
        print(traceback.format_exc())
        raise

async def edge_text_to_speech_with_fallback(text, character="alice"):
    """Version avec fallback vers votre ancien système si Edge échoue"""
    try:
        # Essayer Edge-TTS en premier
        return await edge_text_to_speech(text, character)
    except Exception as edge_error:
        print(f"💀 Edge-TTS complètement échoué: {edge_error}")
        print("🔄 Fallback vers ancien système TTS...")
        
        try:
            # Fallback vers votre ancien système
            from back.tts_utils import text_to_speech_audio
            
            # Mapping vers vos anciennes voix
            voice_mapping = {
                "mike": "english_sam_tacotron-DDC",
                "alice": "english_ljspeech_vits", 
                "sarah": "english_ljspeech_tacotron2-DDC"
            }
            
            voice_key = voice_mapping.get(character, "english_ljspeech_tacotron2-DDC")
            return await text_to_speech_audio(text, voice_key)
            
        except Exception as fallback_error:
            print(f"💀 Même le fallback a échoué: {fallback_error}")
            raise Exception(f"Edge-TTS et fallback ont échoué. Edge: {edge_error}, Fallback: {fallback_error}")

# Test unitaire de la fonction
async def test_edge_tts():
    """Fonction de test pour vérifier que Edge-TTS fonctionne"""
    test_cases = [
        ("alice", "Hello! I'm Alice, your friendly conversation partner."),
        ("mike", "NYC Taxi Central, Mike speaking. Where would you like to go?"),
        ("sarah", "Welcome to our airline. How can I assist you with your booking today?")
    ]
    
    print("🧪 === TEST EDGE-TTS ===")
    
    for character, text in test_cases:
        try:
            print(f"\n🎭 Test {character}...")
            audio_path, duration = await edge_text_to_speech(text, character)
            print(f"✅ {character}: Réussi - {audio_path} ({duration:.2f}s)")
        except Exception as e:
            print(f"❌ {character}: Échoué - {e}")
    
    print("\n🧪 === FIN TEST ===")

if __name__ == "__main__":
    # Test direct si le fichier est exécuté
    asyncio.run(test_edge_tts())