from langdetect.lang_detect_exception import LangDetectException
from langdetect import detect_langs, DetectorFactory
from langid.langid import LanguageIdentifier, model

DetectorFactory.seed = 0

def detect_language(text):
    try:
        langid_identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        langid_identifier.set_languages(['en', 'fr'])
        langid_lang, langid_confidence = langid_identifier.classify(text)
        languages = detect_langs(text)
        filtered_langs = [lang for lang in languages if lang.lang in ['en', 'fr']]
        
        if not filtered_langs:
            langdetect_lang = 'unknown'
            langdetect_confidence = 0
        else:
            best_guess = max(filtered_langs, key=lambda lang: lang.prob)
            langdetect_lang = best_guess.lang
            langdetect_confidence = best_guess.prob
        
        confidence_threshold = 0.3

        print(f"Langid detected: {langid_lang} with confidence {langid_confidence}")
        print(f"Langdetect detected: {langdetect_lang} with confidence {langdetect_confidence}")
        
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
  