import langid
from langdetect.lang_detect_exception import LangDetectException
from langdetect import detect_langs, DetectorFactory

DetectorFactory.seed = 0

def detect_language(text):
    try:
        # Utilisation de langid
        langid.set_languages(['en', 'fr'])  # Définir les langues à considérer
        langid_lang, langid_confidence = langid.classify(text)
        
        # Utilisation de langdetect
        languages = detect_langs(text)
        filtered_langs = [lang for lang in languages if lang.lang in ['en', 'fr']]
        
        if not filtered_langs:
            langdetect_lang = 'unknown'
            langdetect_confidence = 0
        else:
            best_guess = max(filtered_langs, key=lambda lang: lang.prob)
            langdetect_lang = best_guess.lang
            langdetect_confidence = best_guess.prob
        
        # Seuil de confiance
        confidence_threshold = 0.3

        # Afficher les résultats intermédiaires
        print(f"Langid detected: {langid_lang} with confidence {langid_confidence}")
        print(f"Langdetect detected: {langdetect_lang} with confidence {langdetect_confidence}")
        
        # Combinaison des résultats
        if langid_confidence >= confidence_threshold and langdetect_confidence >= confidence_threshold:
            if langid_lang == langdetect_lang:
                return langid_lang
            else:
                return 'unknown'
        elif langid_confidence >= confidence_threshold:
            return langid_lang
        elif langdetect_confidence >= confidence_threshold:
            return langdetect_lang
        else:
            return 'unknown'
    
    except LangDetectException:
        return 'unknown'
  