import nltk
nltk.download('punkt')
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

stopwords_es = set(stopwords.words('spanish'))

def preprocesar_texto(texto):
    # Convertir a minúsculas
    texto = texto.lower()
    
    # Eliminar todo lo que no sean letras (reemplaza por espacio)
    texto = re.sub(r'[^a-záéíóúñü]+', ' ', texto)
    
    # Tokenizar
    tokens = word_tokenize(texto)
    
    # Eliminar stopwords
    tokens = [palabra for palabra in tokens if palabra not in stopwords_es]
    
    return tokens

def obtener_ngrams(tokens, n):
    return list(ngrams(tokens, n))

def calcular_similitud_por_ngramas(tokens1, tokens2, n):
    ngrams1 = obtener_ngrams(tokens1, n)
    ngrams2 = obtener_ngrams(tokens2, n)
    
    set1 = set(ngrams1)
    set2 = set(ngrams2)
    
    interseccion = set1.intersection(set2)
    union = set1.union(set2)

    if not union:
        return 0.0
    
    similitud = len(interseccion) / len(union)
    return similitud * 100  # porcentaje


def riqueza_lexica(tokens):
    total_palabras = len(tokens)
    palabras_unicas = len(set(tokens))
    
    if total_palabras == 0:
        return 0.0
    
    return palabras_unicas / total_palabras
