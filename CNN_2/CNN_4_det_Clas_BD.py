from mtcnn import MTCNN
import cv2
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# Reconstruir el modelo
image_size = (128, 128)  # Tamaño correcto
num_classes = 2  # Hombre y Mujer
weights_path = "model_weights_reco_2030.h5"  # Ruta de los pesos

# Reconstrucción del modelo
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Congelar capas base

model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')  # Capa final de salida con softmax
])

# Cargar los pesos
model.load_weights(weights_path)
print("Modelo cargado correctamente.")

# Mapeo de etiquetas
class_labels = ["Male", "Female"]

# Inicializar el detector de rostros
detector = MTCNN()

# Preprocesamiento de rostros
def preprocess_face(face_img, target_size=(128, 128)):
    face_img = cv2.resize(face_img, target_size)  # Redimensionar
    face_img = face_img / 255.0  # Normalizar a [0, 1]
    mean = np.array([0.5, 0.5, 0.5])  # Ajusta si usaste otro valor
    std = np.array([0.2, 0.2, 0.2])  # Ajusta si usaste otro valor
    face_img = (face_img - mean) / std  # Normalización global
    face_img = np.expand_dims(face_img, axis=0)  # Añadir batch dimension
    return face_img

# Carpetas para procesar
folders = {
    "hombres": "feret_test/hombres",  # Cambia a tu ruta
    "mujeres": "feret_test/mujeres"   # Cambia a tu ruta
}

# Procesar imágenes
for category, folder_path in folders.items():
    print(f"Procesando carpeta: {category}")
    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)
        if not os.path.isfile(image_path):
            continue

        # Leer imagen
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error al leer la imagen: {image_path}")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb_image)

        for result in results:
            x, y, width, height = result['box']
            face = image[y:y + height, x:x + width]  # Recortar el rostro

            # Validar recorte
            if face.size == 0:
                continue

            # Preprocesar y predecir
            preprocessed_face = preprocess_face(face)
            predictions = model.predict(preprocessed_face)
            predicted_class = np.argmax(predictions)
            confidence = predictions[0][predicted_class]

            # Dibujar resultados
            label = f"{class_labels[predicted_class]}: {confidence:.2f}"
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Mostrar la imagen
        cv2.imshow(f"{category}: {file_name}", image)
        cv2.waitKey(1000)  # Mostrar cada imagen por 1 segundo
        cv2.destroyAllWindows()

    print(f"Finalizado procesamiento de la carpeta: {category}")

cv2.destroyAllWindows()
