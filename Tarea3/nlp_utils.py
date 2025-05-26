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
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import re

def summarize_text_portions(text, lenguaje='english', portion_size=5000, num_sentences=2):
    # Elegimos tres partes: inicio, medio, final
    total_length = len(text)
    portions = [
        text[:portion_size],
        text[total_length//2:total_length//2 + portion_size],
        text[-portion_size:]
    ]

    summarizer = LsaSummarizer()
    tokenizer = Tokenizer(lenguaje)

    summary = []

    for part in portions:
        parser = PlaintextParser.from_string(part, tokenizer)
        sentences = summarizer(parser.document, num_sentences)
        summary.extend(sentences)

    return ' '.join(str(s) for s in summary)

def analyze_text_basic(text, lenguaje):
    from nltk import sent_tokenize, word_tokenize

    if lenguaje == 'english':
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
    else:
        sentences = sent_tokenize(text, language='spanish')
        words = word_tokenize(text, language='spanish')

    num_sentences = len(sentences)
    num_words = len(words)

    # Resumen optimizado
    summary = summarize_text_portions(text, lenguaje)

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
    sentence_splitter = re.compile(r'(?<=[.!?,])\s+')
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

def get_qa_pipeline(language):
    if language == 'english':
        return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    else:
        return pipeline("question-answering")
def answer_question(question, text, language):
    qa_pipeline = get_qa_pipeline(language)
    
    # Para evitar errores, cortamos el contexto si es muy largo
    context = text[:1000] if len(text) > 1000 else text

    result = qa_pipeline({
        'context': context,
        'question': question
    })

    return result['answer']