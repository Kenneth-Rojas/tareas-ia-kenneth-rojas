import nltk
nltk.download('gutenberg')
import nlp_utils as nlp
from flask import Flask, render_template, request, redirect, url_for, session
from nltk.corpus import gutenberg
app = Flask(__name__)
app.secret_key = 'pass1234'  # Necesario para usar session

@app.route('/', methods=['GET', 'POST'])
def select_language():
    if request.method == 'POST':
        language = request.form.get('language')
        if language in ['english', 'spanish']:
            session['language'] = language  # Guardamos idioma elegido
            return redirect(url_for('select_book'))
        else:
            return "Idioma no soportado", 400
    return render_template('index.html')

@app.route('/analyze_basic', methods=['POST'])
def analyze_basic():
    text = request.form.get('book_text')  # Esto vendrá de un formulario
    if not text:
        return "No se proporcionó texto para analizar.", 400

    results = nlp.analyze_text_basic(text)
    return render_template('result.html', results=results)
@app.route('/select_book')
def select_book():
    language = session.get('language')
    if not language:
        return redirect(url_for('select_language'))

    # Lista filtrada de libros (ejemplo básico)
    english_books = [fileid for fileid in gutenberg.fileids() if fileid.endswith('.txt')]
    spanish_books = []  # Gutenberg de NLTK no tiene libros en español por defecto

    if language == 'english':
        available_books = english_books
    elif language == 'spanish':
        available_books = spanish_books
    else:
        return "Idioma no soportado", 400

    return render_template('select_book.html', books=available_books, language=language)