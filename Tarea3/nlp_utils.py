import nltk
nltk.download('gutenberg')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')  # Para análisis de sentimientos
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string

def analyze_text_basic(text,lenguaje):
    # Tokenización de oraciones y palabras
    pattern = r'''(?x)                 # set flag to allow verbose regexps
              (?:[A-Z]\.)+         # abbreviations, e.g. U.S.A.
              | \w+(?:-\w+)*       # words with optional internal hyphens
              | \$?\d+(?:\.\d+)?%? # currency and percentages, e.g. $12.40, 82%
              | \.\.\.             # ellipsis
              | [][.,;"'?():-_`]   # these are separate tokens; includes ], ['''
    if lenguaje == 'english':
        sentences = nltk.regexp_tokenize(text, pattern)
        words = nltk.word_tokenize(text.lower())
    else:  # Asumimos español si no es inglés
        sentences = nltk.regexp_tokenize(text, pattern)
        words = nltk.word_tokenize(text, language='spanish')

    
    # Conteo
    num_sentences = len(sentences)
    num_words = len(words)

    # Resumen simple: tomar 3 primeras oraciones como ejemplo
    summary = ' '.join(sentences[:3])

    return {
        'num_sentences': num_sentences,
        'num_words': num_words,
        'summary': summary
    }