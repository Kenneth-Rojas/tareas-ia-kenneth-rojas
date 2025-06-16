import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
iris = load_iris()
X, y = iris.data, iris.target

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

labels = kmeans.labels_

# Visualizar agrupaciones
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("Clustering con KMeans")
plt.xlabel("Largo Sépalo")
plt.ylabel("Ancho Sépalo")
plt.show()
print("Etiquetas reales:", y[:10])
print("Etiquetas clustering:", labels[:10])

from sklearn.metrics import confusion_matrix
import seaborn as sns

conf = confusion_matrix(y, labels)
sns.heatmap(conf, annot=True, cmap='Blues', fmt='d')
plt.xlabel("Clúster asignado por KMeans")
plt.ylabel("Etiqueta real")
plt.title("Matriz de Confusión")
plt.show()

