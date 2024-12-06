import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# Definir el path de la base de datos
dataset_dir = 'feret_train'  # Cambia esto con la ruta a tu base de datos
hombres_dir = os.path.join(dataset_dir, 'hombres_reco')
mujeres_dir = os.path.join(dataset_dir, 'mujeres_reco')

# Configuración
image_size = (128, 128)  # Tamaño al que redimensionar las imágenes
num_classes = 2  # Dos clases: hombres y mujeres
batch_size = 32  # Tamaño de batch

# Función para cargar las imágenes y sus etiquetas
def load_images_from_directory(directory, label, image_size):
    images = []
    labels = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = image.load_img(img_path, target_size=image_size)
        img_array = image.img_to_array(img) / 255.0  # Normalizar a [0, 1]
        images.append(img_array)
        labels.append(label)
    return images, labels

# Cargar imágenes
hombres_images, hombres_labels = load_images_from_directory(hombres_dir, 0, image_size)
mujeres_images, mujeres_labels = load_images_from_directory(mujeres_dir, 1, image_size)

# Combinar las imágenes de ambos directorios
images = np.array(hombres_images + mujeres_images)
labels = np.array(hombres_labels + mujeres_labels)

# Normalizar las imágenes (usando media y desviación estándar)
mean = np.mean(images, axis=(0, 1, 2))
std = np.std(images, axis=(0, 1, 2))
images = (images - mean) / std

# Convertir las etiquetas a one-hot encoding
labels = to_categorical(labels, num_classes)

# Dividir en conjunto de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Calcular pesos para clases desbalanceadas
class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(labels, axis=1)), y=np.argmax(labels, axis=1))
class_weights = dict(enumerate(class_weights))

# Implementar transfer learning usando MobileNetV2
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Congelar capas preentrenadas

# Construir el modelo
model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Resumen del modelo
model.summary()

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=1e-4), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Callbacks para ajustar el learning rate y guardar el mejor modelo
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

# Entrenamiento del modelo
history = model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=batch_size,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=[lr_scheduler, checkpoint]
)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# Guardar los pesos del modelo
weights_path = 'model_weights_reco_200.h5'
model.save_weights(weights_path)
print(f"Pesos guardados en {weights_path}")

# Visualizar predicciones
def plot_predictions(x_test, y_test, model, num_samples=10):
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    
    random_indices = np.random.choice(len(x_test), num_samples, replace=False)
    for index in random_indices:
        plt.figure(figsize=(5, 5))
        plt.imshow(x_test[index])
        predicted_label = predicted_labels[index]
        true_label = true_labels[index]
        plt.title(f"Predicción: {'Mujer' if predicted_label == 1 else 'Hombre'} - Real: {'Mujer' if true_label == 1 else 'Hombre'}")
        plt.axis('off')
        plt.show()

# Visualizar predicciones en imágenes de prueba
plot_predictions(x_test, y_test, model, num_samples=10)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Obtener las etiquetas reales y predichas
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(x_test), axis=1)

# Generar la matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred)

# Normalizar la matriz de confusión a porcentaje
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

# Crear el gráfico de la matriz de confusión
fig, ax = plt.subplots(figsize=(5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_percent, display_labels=['Hombre', 'Mujer'])
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format=".2f")
plt.title("Matriz de Confusión (en porcentajes)")

# Guardar la imagen
output_path = "matriz_confusion_porcentaje.png"
plt.savefig(output_path)
print(f"Matriz de confusión guardada en: {output_path}")

# Mostrar la gráfica
plt.show()

