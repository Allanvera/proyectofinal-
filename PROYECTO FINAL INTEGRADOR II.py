# Paso 1: Librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Paso 2: Dataset simulado
data = {
    'n_opiniones': [15, 2, 50, 40, 5, 60, 3, 25],
    'calificacion_promedio': [4.5, 2.1, 4.8, 4.2, 1.9, 4.9, 2.5, 3.9],
    'reportes': [0, 3, 1, 0, 4, 0, 2, 0],
    'verificado': [1, 0, 1, 1, 0, 1, 0, 1],
    'es_confiable': [1, 0, 1, 1, 0, 1, 0, 1]  # Etiqueta
}

df = pd.DataFrame(data)

# Paso 3: División en entrenamiento y prueba
X = df[['n_opiniones', 'calificacion_promedio', 'reportes', 'verificado']]
y = df['es_confiable']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Paso 4: Entrenamiento del modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Paso 5: Validación del modelo
y_pred = modelo.predict(X_test)
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Paso 6: Prueba con nuevo usuario
nuevo_usuario = pd.DataFrame({
    'n_opiniones': [10],
    'calificacion_promedio': [4.3],
    'reportes': [0],
    'verificado': [1]
})
resultado = modelo.predict(nuevo_usuario)
print("¿Usuario confiable?", "Sí" if resultado[0] == 1 else "No")
