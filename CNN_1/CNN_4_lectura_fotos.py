import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Definir el path de la base de datos
dataset_dir = 'feret'  # Cambia esto con la ruta a tu base de datos
hombres_dir = os.path.join(dataset_dir, 'hombres')
mujeres_dir = os.path.join(dataset_dir, 'mujeres')

# Configuración
image_size = (128, 128)  # Tamaño al que redimensionar las imágenes
num_classes = 2  # Dos clases: hombres y mujeres

# Función para cargar las imágenes y sus etiquetas
def load_images_from_directory(directory, label, image_size):
    images = []
    labels = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = image.load_img(img_path, target_size=image_size)  # Cargar la imagen y redimensionarla
        img_array = image.img_to_array(img) / 255.0  # Convertir a array y normalizar a [0, 1]
        images.append(img_array)
        labels.append(label)  # Añadir la etiqueta correspondiente (0 para hombres, 1 para mujeres)
    return images, labels

# Cargar imágenes
hombres_images, hombres_labels = load_images_from_directory(hombres_dir, 0, image_size)
mujeres_images, mujeres_labels = load_images_from_directory(mujeres_dir, 1, image_size)

# Combinar las imágenes de ambos directorios
images = np.array(hombres_images + mujeres_images)
labels = np.array(hombres_labels + mujeres_labels)

# Convertir las etiquetas a one-hot encoding
labels = to_categorical(labels, num_classes)

# Cargar el modelo con los pesos guardados
model = Sequential()

# Definir la arquitectura del modelo
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3), padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(num_classes, activation='softmax'))  # softmax para clasificación multiclase

# Cargar los pesos del archivo 'model_weights.h5'
model.load_weights('model_weights.h5')

# Hacer predicciones en todas las imágenes
predictions = model.predict(images)

# Convertir las predicciones a clases
predicted_labels = np.argmax(predictions, axis=1)

# Encontrar los índices de los hombres que fueron clasificados como mujeres (predicción = 1 y etiqueta real = 0)
hombres_clasificados_como_mujeres = np.where((predicted_labels == 1) & (np.argmax(labels, axis=1) == 0))[0]

# Encontrar los índices de las mujeres que fueron clasificados como hombres (predicción = 0 y etiqueta real = 1)
mujeres_clasificados_como_hombres = np.where((predicted_labels == 0) & (np.argmax(labels, axis=1) == 1))[0]

# Mostrar las imágenes de hombres clasificados como mujeres
print("Hombres clasificados como mujeres:")
for index in hombres_clasificados_como_mujeres:
    plt.figure(figsize=(5, 5))
    plt.imshow(images[index])
    plt.title(f"Predicción: Mujer - Real: Hombre")
    plt.axis('off')
    plt.show()

# Mostrar las imágenes de mujeres clasificados como hombres
print("Mujeres clasificados como hombres:")
for index in mujeres_clasificados_como_hombres:
    plt.figure(figsize=(5, 5))
    plt.imshow(images[index])
    plt.title(f"Predicción: Hombre - Real: Mujer")
    plt.axis('off')
    plt.show()
