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
        sentence_count=result['num_sentences'],
        language=language
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
    book = session.get('book')
    idioma = session.get('language')
    if not book or not idioma:
        return redirect('/')
    if idioma == 'english':
        text = gutenberg.raw(book)
    else:
        path = os.path.join('spanish_books', book)
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
    
    result = nlp.analyze_sentiment(text, idioma)

    return render_template('sentiment_result.html', result=result,language=idioma)

@app.route('/entity_analysis', methods=['POST'])
def entity_analysis():
    book = session.get('book')
    language = session.get('language')
    if not book or not language:
        return redirect('/')

    if language == 'english':
        text = gutenberg.raw(book)
    else:
        with open(os.path.join('spanish_books', book), encoding='utf-8') as f:
            text = f.read()

    result = nlp.extract_topics_entities(text, language)
    return render_template('entities_result.html', result=result, language=language)

@app.route('/flashcards', methods=['POST'])
def flashcards():
    book = session.get('book')
    language = session.get('language')
    if not book or not language:
        return redirect('/')

    if language == 'english':
        text = gutenberg.raw(book)
    else:
        with open(os.path.join('spanish_books', book), encoding='utf-8') as f:
            text = f.read()

    result = nlp.generate_flashcards(text, language)
    return render_template('flashcards.html', flashcards=result, language=language)

@app.route('/qa', methods=['GET', 'POST'])
def qa():
    language = session.get('language')
    book = session.get('book')

    if not language or not book:
        return redirect('/')

    if language == 'english':
        raw_text = gutenberg.raw(book)
    else:
        with open(os.path.join("spanish_books", book), 'r', encoding='utf-8') as f:
            raw_text = f.read()

    answer = None
    if request.method == 'POST':
        question = request.form.get('question', '')
        if question:
            answer = nlp.answer_question(question, raw_text, language)

    return render_template('qa.html', answer=answer, language=language)




if __name__ == '__main__':
    app.run(debug=True)

