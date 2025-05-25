import nltk
nltk.download('gutenberg')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')  # Para análisis de sentimientos
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string

def analyze_text_basic(text):
    # Tokenización de oraciones y palabras
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    # Filtrar puntuación y stopwords
    stop_words = set(stopwords.words('english'))  # o 'spanish' según idioma
    words_cleaned = [w for w in words if w.isalpha() and w not in stop_words]

    # Conteo
    num_sentences = len(sentences)
    num_words = len(words_cleaned)

    # Resumen simple: tomar 3 primeras oraciones como ejemplo
    summary = ' '.join(sentences[:3])

    return {
        'num_sentences': num_sentences,
        'num_words': num_words,
        'summary': summary
    }