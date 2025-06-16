import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Cargar datos
df = pd.read_csv("Auto.csv")

# Convertir 'horsepower' a numérico y eliminar filas con NaNs
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df = df.dropna()

# Seleccionar variables predictoras y objetivo
X = df[['displacement', 'cylinders', 'weight']]
y = df['mpg']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predicción
y_pred = modelo.predict(X_test)

# Métricas
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Gráfico
plt.scatter(y_test, y_pred)
plt.xlabel("MPG Real")
plt.ylabel("MPG Predicho")
plt.title("Regresión Lineal: MPG vs Cilindrada, Cilindros y Peso")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.grid(True)
plt.show()
