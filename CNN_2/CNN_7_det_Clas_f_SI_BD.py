from mtcnn import MTCNN
import cv2
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Reconstruir el modelo
image_size = (128, 128)
num_classes = 2
weights_path = "model_weights_reco_200.h5"

# Reconstrucción del modelo
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
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
    face_img = cv2.resize(face_img, target_size)
    face_img = face_img / 255.0
    mean = np.array([0.5, 0.5, 0.5])  # Ajusta si usaste otros valores
    std = np.array([0.2, 0.2, 0.2])  # Ajusta si usaste otros valores
    face_img = (face_img - mean) / std
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

# Carpetas para procesar
folders = {
    "hombres": {"path": "feret_test/hombres", "label": 0},  # Cambia a tu ruta
    "mujeres": {"path": "feret_test/mujeres", "label": 1}   # Cambia a tu ruta
}

# Variables para la matriz de confusión
true_labels = []
predicted_labels = []

# Procesar imágenes
for category, info in folders.items():
    folder_path = info["path"]
    true_label = info["label"]

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
            face = image[y:y + height, x:x + width]

            # Validar recorte
            if face.size == 0:
                continue

            # Preprocesar y predecir
            preprocessed_face = preprocess_face(face)
            predictions = model.predict(preprocessed_face)
            predicted_class = np.argmax(predictions)

            # Guardar etiquetas reales y predichas
            true_labels.append(true_label)
            predicted_labels.append(predicted_class)

# Mostrar matriz de confusión con porcentajes
conf_matrix = confusion_matrix(true_labels, predicted_labels)
conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

print("\nMatriz de Confusión (Porcentajes):")
print(f"{'':>10}{'Male':>10}{'Female':>10}")
for i, label in enumerate(class_labels):
    print(f"{label:>10}{conf_matrix_percentage[i, 0]:>10.2f}{conf_matrix_percentage[i, 1]:>10.2f}")

# Crear un gráfico de la matriz de confusión sin `sns`
fig, ax = plt.subplots(figsize=(8, 6))
ax.matshow(conf_matrix_percentage, cmap="Blues", alpha=0.8)

# Anotaciones en cada celda
for i in range(conf_matrix_percentage.shape[0]):
    for j in range(conf_matrix_percentage.shape[1]):
        ax.text(
            x=j, y=i,
            s=f"{conf_matrix_percentage[i, j]:.2f}%",
            va='center', ha='center',
            color="black"
        )

# Etiquetas y diseño
ax.set_xticks(range(len(class_labels)))
ax.set_yticks(range(len(class_labels)))
ax.set_xticklabels(class_labels, fontsize=12)
ax.set_yticklabels(class_labels, fontsize=12)
ax.set_xlabel("Etiqueta Predicha", fontsize=14)
ax.set_ylabel("Etiqueta Real", fontsize=14)
ax.set_title("Matriz de Confusión (Porcentajes)", fontsize=16)

plt.show()
