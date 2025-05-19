import matplotlib.pyplot as plt
from collections import Counter

def graficar_palabras_mas_frecuentes(tokens1, tokens2, top=15):
    # Unimos los tokens de ambos textos
    tokens_totales = tokens1 + tokens2
    
    # Contamos frecuencia
    frecuencia = Counter(tokens_totales)
    
    # Seleccionamos las más comunes
    palabras_comunes = frecuencia.most_common(top)
    
    palabras, conteos = zip(*palabras_comunes)

    # Gráfica
    plt.figure(figsize=(10, 5))
    plt.bar(palabras, conteos, color='skyblue')
    plt.xticks(rotation=45)
    plt.title(f'Top {top} palabras más frecuentes en ambos textos')
    plt.xlabel('Palabras')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()

