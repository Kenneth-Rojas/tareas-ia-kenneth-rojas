import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('gutenberg')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')  # Para análisis de sentimientos
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import spacy
from collections import Counter
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import re

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
def analyze_sentiment(text, language):
    if language == 'english':
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        sentiment = 'Positive' if scores['compound'] > 0.05 else 'Negative' if scores['compound'] < -0.05 else 'Neutral'
        return {
            'sentiment': sentiment,
            'scores': scores
        }
    else:  # español
        sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
        result = sentiment_pipeline(text[:512])[0]  # Limitamos texto largo
        return {
            'sentiment': result['label'],
            'score': result['score']
        }
def extract_topics_entities(text, language):
    if language == 'english':
        nlp = spacy.load("en_core_web_sm")
    else:
        nlp = spacy.load("es_core_news_sm")
    nlp.max_length = 2_500_000  # Aumentar el límite de longitud del texto
    paragraphs = text.split("\n\n")  # o usar otra lógica de división
    entity_counter = Counter()
    noun_counter = Counter()

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        try:
            doc = nlp(para)
            if language == 'english':
                entity_labels = ("PERSON", "ORG", "GPE")
            else:
                entity_labels = ("PER", "ORG", "LOC")

            # Entidades nombradas
            entities = [ent.text for ent in doc.ents if ent.label_ in entity_labels]
            entity_counter.update(entities)

            # Sustantivos como temas
            nouns = [token.lemma_.lower() for token in doc if token.pos_ == "NOUN" and not token.is_stop]
            noun_counter.update(nouns)

        except Exception as e:
            print("Error procesando fragmento:", e)
            continue  # salta fragmentos problemáticos

    return {
        "entities": entity_counter.most_common(10),
        "topics": noun_counter.most_common(10)
    }
def split_paragraphs(text, min_length=300):
    paragraphs = [p for p in text.split('\n') if len(p.strip()) > min_length]
    return ' '.join(paragraphs[:5])  # Solo usar los primeros 5 párrafos largos

def generate_flashcards(text, language):
    # Tokenización básica para dividir en palabras y frases
    sentence_splitter = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_splitter.split(text)
    text_short = ' '.join(sentences[:50])  # Solo las primeras 50 oraciones

    # Extraer sustantivos como temas (usando heurística por idioma)
    if language == 'english':
        nouns = re.findall(r'\b(?:[A-Z][a-z]+|[a-z]{4,})\b', text_short)
    else:
        nouns = re.findall(r'\b(?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+|[a-záéíóúñ]{4,})\b', text_short)

    common_nouns = [word.lower() for word, _ in Counter(nouns).most_common(20)]

    # Usar sumy para generar resumen
    parser = PlaintextParser.from_string(text_short, Tokenizer(language))
    summarizer = TextRankSummarizer()
    summary_sentences = summarizer(parser.document, 10)

    # Generar flashcards
    flashcards = []
    for i, noun in enumerate(common_nouns[:10]):
        summary = str(summary_sentences[i]) if i < len(summary_sentences) else "Resumen no disponible."
        flashcards.append((noun.capitalize(), summary))

    return flashcards