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
dataset_dir = 'C:/FotosDocentes/'  # Cambia esto con la ruta a tu base de datos
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

# Crear la figura para mostrar las imágenes
def mostrar_imagenes_mal_clasificadas(indices_mal_clasificados, x_data, pred_labels, y_data, title):
    n_images = len(indices_mal_clasificados)
    ncols = 5
    nrows = (n_images // ncols) + (n_images % ncols > 0)  # Cálculo de filas necesarias

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows))
    axes = axes.flatten()

    for i in range(n_images):
        index = indices_mal_clasificados[i]
        ax = axes[i]
        ax.imshow(x_data[index])
        true_label = np.argmax(y_data[index])
        predicted_label = pred_labels[index]
        ax.set_title(f"Pred: {'Mujer' if predicted_label == 1 else 'Hombre'}\nReal: {'Mujer' if true_label == 1 else 'Hombre'}", fontsize=8)
        ax.axis('off')

    # Eliminar los ejes sobrantes si hay menos imágenes que espacios en la cuadrícula
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# Mostrar hombres mal clasificados como mujeres
mostrar_imagenes_mal_clasificadas(hombres_clasificados_como_mujeres, images, predicted_labels, labels, "Hombres clasificados como mujeres")

# Mostrar mujeres mal clasificados como hombres
mostrar_imagenes_mal_clasificadas(mujeres_clasificados_como_hombres, images, predicted_labels, labels, "Mujeres clasificados como hombres")

# Calcular la matriz de confusión
cm = confusion_matrix(np.argmax(labels, axis=1), predicted_labels)

# Normalizar la matriz de confusión para obtener porcentajes
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Visualizar la matriz de confusión con porcentajes
plt.figure(figsize=(5, 5))
plt.imshow(cm_percentage, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión con Porcentajes')
plt.colorbar()

# Etiquetas de los ejes
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, ['Hombres', 'Mujeres'])
plt.yticks(tick_marks, ['Hombres', 'Mujeres'])

# Añadir los valores de porcentaje sobre la matriz
thresh = cm_percentage.max() / 2.
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, f'{cm_percentage[i, j]:.2f}%', ha="center", va="center", 
                 color="white" if cm_percentage[i, j] > thresh else "black")

# Etiquetas de los ejes
plt.xlabel('Predicción')
plt.ylabel('Etiqueta Real')
plt.show()
