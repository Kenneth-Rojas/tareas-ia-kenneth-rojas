# Proyecto de PLN
Este proyecto toma 2 archivos de texto (no escaneado) y las compara atraves de n-gramas, se calcula su porcentaje de similitud, la riqueza lexica de cada uno de los textos y se muestra una grafica de barras con las palabras mas utilizadas en ambos textos.
# Ejecucion
Antes de ejecutar el proyecto es importante verificar si se cuenta con las siguientes librerias:
-pymupdf
-nltk
-matplotlib
Adicionalmente se debe cambiar las lineas 5 y 6 de main.py , estas corresponen a las rutas de los archivos pdf a comparar.
Se debe cambiar la linea 8 si se desea usar n-gramas diferentes a 2 (default).
Una vez se hayan realizado los cambios y configuraciones necesarias se puede ejecutar el codigo
python main.py
# Comentarios
Codigo generado y depurado con ayuda de ChatGPT: https://chatgpt.com/share/682adbe9-1290-8002-bce1-2a2fa05eb92c