##**1. Importación de librerias**
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
import matplotlib.pyplot as plt

"""##**2. Data set**"""
Cargando el data set de UC Irvine Machine Learning Repository con el nombre de **SMS Spam Collection.**
- https://archive.ics.uci.edu/dataset/228/sms+spam+collection

# Cargar el dataset de mensajes
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/INF354-Inteligencia Artificial/Exposicion/datasets/SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])

# Visualizar las primeras filas
print(data.head())

# Asumimos que el dataset tiene dos columnas: 'text' y 'label'
# 'text' es el mensaje y 'label' es 1 para spam y 0 para no spam
X = data['message']  # Los mensajes de texto
y = data['label']  # Las etiquetas de spam (1) y no spam (0)

"""##**2. Conversión de datos**"""
# Convertir los mensajes de texto en vectores de características numéricas
vectorizer = CountVectorizer(stop_words='english')  # Quitamos las palabras comunes (stop words)
X_vectorized = vectorizer.fit_transform(X)

# Visualizamos las primeras 5 características (palabras)
print(vectorizer.get_feature_names_out()[:5])

"""##**3. Entrenamiento del modelo**"""
# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

"""##**4. Clasificador de Bayes**"""
# Entrenar el clasificador Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

"""##**5. Precisión y Evaluación del modelo**"""
# Predicciones del modelo
y_pred = model.predict(X_test)

# Calcular precisión, recall y exactitud
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, pos_label='spam')
precision = precision_score(y_test, y_pred, pos_label='spam')

# Mostrar resultados
print(f"Exactitud: {accuracy * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")

"""##**6. Matriz de confusión**"""
# Generar la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Mostrar la matriz de confusión
print("Matriz de confusión:")
print(cm)

"""##**7. Grafica de precision y recall (o sensibilida)**"""
# Graficar la precisión y recall
labels = ['Precision', 'Recall']
scores = [precision * 100, recall * 100]
plt.bar(labels, scores, color=['blue', 'orange'])
plt.ylabel('Percentage (%)')
plt.title('Precision and Recall of the Naive Bayes Classifier')
plt.show()