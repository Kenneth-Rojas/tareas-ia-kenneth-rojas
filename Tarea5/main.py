import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import shap
from lime.lime_tabular import LimeTabularExplainer

# Cargar y preparar datos
df = pd.read_csv("credit_data.csv")
label_encoder = LabelEncoder()
df['Historial Crediticio'] = label_encoder.fit_transform(df['Historial Crediticio'])
df['Historial de Pagos'] = label_encoder.fit_transform(df['Historial de Pagos'])
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f'Accuracy: {accuracy_score(y_test, model.predict(X_test))}')

# Crear explainer SHAP
explainer = shap.Explainer(model, X_train, model_output="probability")

# Obtener explicaciones
shap_values = explainer(X_test)
print("shap_values.values.shape:", shap_values.values.shape)

# Extraer shap_values para clase 1 (clase positiva)
class1_shap_values = shap_values.values[:, :, 1]
class1_base_values = shap_values.base_values[:, 1]

# Crear objeto Explanation para beeswarm plot
exp_class1 = shap.Explanation(
    values=class1_shap_values,
    base_values=class1_base_values,
    data=X_test.values,
    feature_names=X_test.columns.tolist()
)

# Plot beeswarm para clase 1
shap.plots.beeswarm(exp_class1)

# Force plot para la primera muestra (índice 0) usando matplotlib (VSCode no soporta HTML/JS)
shap.plots.force(
    class1_base_values[0],
    class1_shap_values[0],
    X_test.iloc[0],
    matplotlib=True
)

# === LIME ===
lime_explainer = LimeTabularExplainer(
    X_train.values,
    training_labels=y_train.values,
    feature_names=X_train.columns.tolist(),
    mode="classification"
)

test_instance = X_test.iloc[0].values
lime_exp = lime_explainer.explain_instance(test_instance, model.predict_proba)

# Guardar explicación LIME en HTML
with open("lime_exp.html", "w") as f:
    f.write(lime_exp.as_html())
print("Explicación LIME guardada en lime_exp.html, ábrelo en tu navegador.")
