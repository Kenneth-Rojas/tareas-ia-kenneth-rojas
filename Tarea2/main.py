import nlp as nlp
import pdf as pdf
import plot as plot
# Reemplazar rutas
ruta1 = '/home/kenny/Downloads/Prerreporte3Digitales.pdf'
ruta2 = '/home/kenny/Downloads/Prerreporte3Digitales.pdf'
# ngramas
n = 2
# Funciones de procesamiento de texto
texto1 = pdf.extraer_texto_pdf(ruta1)
texto2 = pdf.extraer_texto_pdf(ruta2)
tokens1 = nlp.preprocesar_texto(texto1)
tokens2 = nlp.preprocesar_texto(texto2)
# Calcular similitud
similitud_bigramas = nlp.calcular_similitud_por_ngramas(tokens1, tokens2, n)
# Calcular riqueza léxica
riqueza1 = nlp.riqueza_lexica(tokens1)
riqueza2 = nlp.riqueza_lexica(tokens2)
# Resultados
print(f"Similitud por bigramas: {similitud_bigramas:.2f}%")
print(f"Riqueza léxica texto 1: {riqueza1:.2f}")
print(f"Riqueza léxica texto 2: {riqueza2:.2f}")
# Graficar palabras más frecuentes
plot.graficar_palabras_mas_frecuentes(tokens1, tokens2, top=15)

