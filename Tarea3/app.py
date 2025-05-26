import nltk
nltk.download('gutenberg')
import nlp_utils as nlp
from flask import Flask, render_template, request, redirect, url_for, session
from nltk.corpus import gutenberg
import os
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
    language = session.get('language')
    book = session.get('book')

    if language == 'english':
        raw_text = gutenberg.raw(book)
    elif language == 'spanish':
        with open(f"spanish_books/{book}", 'r', encoding='utf-8') as f:
            raw_text = f.read()
    else:
        return "Idioma no soportado", 400

    result = nlp.analyze_text_basic(raw_text, language)
    return render_template(
        'result.html',
        summary=result['summary'],
        word_count=result['num_words'],
        sentence_count=result['num_sentences']
    )
    
@app.route('/select_book')
def select_book():
    language = session.get('language')
    if not language:
        return redirect(url_for('select_language'))

    # Lista filtrada de libros (ejemplo básico)
    english_books = [fileid for fileid in gutenberg.fileids() if fileid.endswith('.txt')]
    spanish_dir = 'spanish_books'
    spanish_books = [f for f in os.listdir(spanish_dir) if f.endswith('.txt')]

    if language == 'english':
        available_books = english_books
    elif language == 'spanish':
        available_books = spanish_books
    else:
        return "Idioma no soportado", 400

    return render_template('select_book.html', books=available_books, language=language)
@app.route('/analysis_menu', methods=['POST'])
def analysis_menu():
    if request.method == 'POST':
       selected_book = request.form['book']
       session['book'] = selected_book
    book = session.get('book')
    language = session.get('language')
    return render_template('analysis_menu.html', book=book, language=language)

@app.route('/sentiment_analysis', methods=['POST'])
def sentiment_analysis():
    return "Análisis de sentimientos aún no implementado."

@app.route('/entity_analysis', methods=['POST'])
def entity_analysis():
    return "Extracción de temas y personajes aún no implementada."

@app.route('/flashcards', methods=['POST'])
def flashcards():
    return "Generación de flashcards aún no implementada."

@app.route('/qa', methods=['POST'])
def qa():
    return "Sistema de preguntas y respuestas aún no implementado."



if __name__ == '__main__':
    app.run(debug=True)

