import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from mtcnn import MTCNN

# Configuración
image_size = (128, 128)
num_classes = 2
weights_path = "model_weights_reco_200.h5"
class_labels = ["Male", "Female"]

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

# Inicializar listas para las etiquetas y probabilidades
all_true_labels = []
all_predicted_probs = []
male_misclassified = []
female_misclassified = []

# Procesar imágenes
for category, info in folders.items():
    folder_path = info["path"]
    true_label = info["label"]

    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)
        if not os.path.isfile(image_path):
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb_image)

        for result in results:
            x, y, width, height = result['box']
            face = image[y:y + height, x:x + width]
            if face.size == 0:
                continue

            preprocessed_face = preprocess_face(face)
            predictions = model.predict(preprocessed_face)
            predicted_class = np.argmax(predictions)
            predicted_prob = predictions[0][1]  # Probabilidad de la clase 'Female'

            all_true_labels.append(true_label)
            all_predicted_probs.append(predicted_prob)

            # Verificar si la predicción es incorrecta
            if predicted_class != true_label:
                label = f"Pred: {class_labels[predicted_class]} ({predicted_prob:.2f}), Real: {class_labels[true_label]}"
                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 2)  # Rectángulo rojo
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if true_label == 0:  # Male misclassified as Female
                    male_misclassified.append(image)
                else:  # Female misclassified as Male
                    female_misclassified.append(image)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(all_true_labels, np.array(all_predicted_probs) > 0.5)
conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

# Mostrar matriz de confusión sin usar sns
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix_percentage, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Matriz de Confusión (Porcentajes)")
plt.colorbar()

# Etiquetas de la matriz de confusión
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels)
plt.yticks(tick_marks, class_labels)

# Colocar los valores de la matriz
thresh = conf_matrix_percentage.max() / 2.
for i in range(conf_matrix_percentage.shape[0]):
    for j in range(conf_matrix_percentage.shape[1]):
        plt.text(j, i, f"{conf_matrix_percentage[i, j]:.2f}%",
                 horizontalalignment="center",
                 color="white" if conf_matrix_percentage[i, j] > thresh else "black")

plt.xlabel("Etiqueta Predicha")
plt.ylabel("Etiqueta Real")
plt.show()

# Mostrar curva ROC
fpr, tpr, thresholds = roc_curve(all_true_labels, all_predicted_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Curva ROC')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.legend(loc='lower right')
plt.show()

# Mostrar imágenes mal clasificadas
def plot_misclassified_images(images, title):
    plt.figure(figsize=(15, 10))
    n_cols = 5
    n_rows = (len(images) // n_cols) + (1 if len(images) % n_cols != 0 else 0)
    
    for i, img in enumerate(images):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# Mostrar imágenes mal clasificadas de Male y Female
plot_misclassified_images(male_misclassified, "Mal Clasificadas - Male Predichas como Female")
plot_misclassified_images(female_misclassified, "Mal Clasificadas - Female Predichas como Male")
