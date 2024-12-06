import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from mtcnn import MTCNN

# Ruta a la carpeta de fotos docentes
dataset_dir = 'Feret_train/'

# Directorios de Hombres y Mujeres
hombres_dir = os.path.join(dataset_dir, 'Hombres')
mujeres_dir = os.path.join(dataset_dir, 'Mujeres')

# Función para cargar y procesar todas las imágenes de una carpeta
def load_and_process_images_from_directory(directory, target_width=600):
    images = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(img_path)
            if img is not None:
                height, width = img.shape[:2]
                aspect_ratio = height / width
                target_height = int(target_width * aspect_ratio)
                resized_image = cv2.resize(img, (target_width, target_height))
                images.append(resized_image)
    return images

# Cargar imágenes de ambas carpetas
hombres_images = load_and_process_images_from_directory(hombres_dir)
mujeres_images = load_and_process_images_from_directory(mujeres_dir)

# Combinar las imágenes y sus etiquetas
all_images = hombres_images + mujeres_images
y_true = [0] * len(hombres_images) + [1] * len(mujeres_images)

# Inicializar el detector de rostros MTCNN
detector = MTCNN()

# Cargar el modelo con los pesos
def load_model_with_weights():
    base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])
    model.load_weights('model_weights_reco_200.h5')
    return model

model = load_model_with_weights()

# Función para detectar rostros y predecir género
def detect_faces_and_predict_gender(input_image, true_label):
    rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_image)
    for result in results:
        x, y, width, height = result['box']
        cv2.rectangle(input_image, (x, y), (x + width, y + height), (0, 255, 255), 4)
        face = input_image[y:y + height, x:x + width]
        face_resized = cv2.resize(face, (128, 128))
        face_array = keras_image.img_to_array(face_resized)
        face_array = np.expand_dims(face_array, axis=0)
        face_array = preprocess_input(face_array)
        prediction = model.predict(face_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        label = 'Female' if predicted_class == 1 else 'Male'
        cv2.putText(input_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Generar subgráficos separados para hombres y mujeres
def plot_images_by_gender(images, labels, gender, rows=5, cols=7):
    # Filtrar las imágenes por género
    gender_images = [img for img, lbl in zip(images, labels) if lbl == gender]
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    axes = axes.ravel()  # Convertir en un array plano
    for idx in range(rows * cols):
        if idx < len(gender_images):
            processed_image = detect_faces_and_predict_gender(gender_images[idx], 'Female' if gender == 1 else 'Male')
            axes[idx].imshow(processed_image)
            axes[idx].axis('off')
        else:
            axes[idx].axis('off')
    plt.tight_layout()
    title = 'Mujeres' if gender == 1 else 'Hombres'
    plt.suptitle(f'Imágenes de {title}', fontsize=20, y=1.02)
    plt.show()

# Llamar a la función de graficado para hombres y mujeres
plot_images_by_gender(all_images, y_true, gender=0, rows=5, cols=10)  # Hombres
plot_images_by_gender(all_images, y_true, gender=1, rows=5, cols=10)  # Mujeres