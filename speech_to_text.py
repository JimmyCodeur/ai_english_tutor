from pocketsphinx import LiveSpeech

def speech_to_text():
    # Utilisation de CMU Sphinx pour la reconnaissance vocale
    speech = LiveSpeech()
    print("Dites quelque chose...")
    for phrase in speech:
        print("Transcription: " + str(phrase))

speech_to_text()