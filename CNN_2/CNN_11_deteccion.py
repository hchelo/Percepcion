import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from mtcnn import MTCNN
import matplotlib.pyplot as plt

# Configuración del modelo
image_size = (128, 128)
class_labels = ["Male", "Female"]
weights_path = "model_weights_reco_1400.h5"
threshold = 0.56  # Umbral del 65%

# Reconstrucción del modelo
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_labels), activation='softmax')
])

# Cargar los pesos del modelo
model.load_weights(weights_path)
print("Modelo cargado correctamente.")

# Inicializar el detector de rostros
detector = MTCNN()

# Preprocesamiento de rostros
def preprocess_face(face_img, target_size=(128, 128)):
    face_img = cv2.resize(face_img, target_size)
    face_img = face_img / 255.0
    mean = np.mean(face_img, axis=(0, 1, 2))
    std = np.std(face_img, axis=(0, 1, 2))
    face_img = (face_img - mean) / std
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

# Leer la imagen con múltiples rostros
input_image_path = "Genero/presentacion.png"  # Cambia a la ruta de tu imagen
image = cv2.imread(input_image_path)
if image is None:
    print("Error: No se pudo leer la imagen.")
else:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detectar rostros
    results = detector.detect_faces(rgb_image)

    for result in results:
        x, y, width, height = result['box']

        # Recortar y preprocesar el rostro
        face = rgb_image[y:y + height, x:x + width]
        if face.size == 0:
            continue
        preprocessed_face = preprocess_face(face)

        # Predecir el género
        predictions = model.predict(preprocessed_face)
        predicted_class = np.argmax(predictions)
        predicted_prob = predictions[0][predicted_class]

        # Verificar si la probabilidad cumple con el umbral
        if predicted_prob >= threshold:
            gender_label = f"{class_labels[predicted_class]} ({predicted_prob:.2f})"

            # Configurar el color para cada género
            if predicted_class == 0:  # Male
                color = (0, 255, 0)  # Verde
            else:  # Female
                color = (255, 0, 255)  # Lila

            # Dibujar el rectángulo y el texto en la imagen original
            cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
            cv2.putText(image, gender_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            print(f"Rostro ignorado: Predicción {class_labels[predicted_class]} con probabilidad {predicted_prob:.2f}")

    # Mostrar la imagen resultante con anotaciones
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Detección de Rostros y Predicción de Género (Umbral > {threshold})")
    plt.show()
