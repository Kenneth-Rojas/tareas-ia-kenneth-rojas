# Proyecto de flask con funciones de PLN
# Ejecucion
Para ejecutar el programa debemos asegurarnos de contar con las siguientes librerias y modelos
pip install flask nltk spacy transformers regex requests beautifulsoup4 pytextrank matplotlib torch
Luego, sobre el directorio Tarea3 ejecutar el siguiente comando:
flask run
Una vez ejecutado se descargara lo necesario para la correcta ejecución de los programas, se debe ingresar al enlace mostrado en la terminal
http://127.0.0.1:5000/
# Descripcion
En el inicio del programa se debe seleccionar el idioma en el que se va a trabajar, luego se decide el libro a analizar.
El programa muestra diferentes funciones:
Resumen y estadisticas: Genera un resumen del libro y cuenta sus oraciones y palabras
Analisis del sentimiento: Analiza el sentimiento general del libro
Tema principal y protagonistas: Muestra los personajes principales y temas relacionados al libro
Flashcards: Genera diferentes tarjetas con subtemas y resumenes utiles a partir del libro
Preguntas y respuestas: Puedes escribir una pregunta que la app intentara responder
# Comentarios
Codigo generado y depurado con ChatGPT: https://chatgpt.com/share/6834f7db-e2e4-8002-927b-4e1e2f0a2dc6
