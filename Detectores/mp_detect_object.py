import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt

# Configuración de visualización de Matplotlib
plt.rcParams['figure.figsize'] = [18, 8]  # Incrementar tamaño de gráficos

# Ruta al modelo
MODEL_PATH = "models/efficientdet_lite0_int8.tflite"
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# Configuración de visualización de bounding boxes
MARGIN = 10  # píxeles
ROW_SIZE = 10  # píxeles
FONT_SIZE = 2
FONT_THICKNESS = 2
TEXT_COLOR = (0, 255, 255)  # amarillo

def visualize(image: np.ndarray, detection_result) -> np.ndarray:
    """Dibuja las cajas delimitadoras y etiquetas en la imagen."""
    for detection in detection_result.detections:
        # Coordenadas de la caja delimitadora
        bbox = detection.bounding_box
        start_point = int(bbox.origin_x), int(bbox.origin_y)
        end_point = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 4)

        # Etiqueta y probabilidad
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f"{category_name} ({probability})"
        text_location = (int(MARGIN + bbox.origin_x), int(MARGIN + ROW_SIZE + bbox.origin_y))
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image

# Cargar imagen
image_path = "city.jpg"
original_image = cv2.imread(image_path)
if original_image is None:
    raise FileNotFoundError(f"No se encontró la imagen en la ruta: {image_path}")

# Convertir a formato RGB para MediaPipe
rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Crear una imagen de MediaPipe
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

# Realizar detección
start_time = time.time()
detection_result = detector.detect(mp_image)
inference_time = (time.time() - start_time) * 1000
print(f"Inference time: {inference_time:.2f} ms")

# Visualizar resultados
annotated_image = visualize(original_image.copy(), detection_result)

# Mostrar imagen original y anotada
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title("Imagen Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.title("Detecciones")
plt.axis("off")

plt.show()

# Guardar resultado
output_path = "city_detected.jpg"
cv2.imwrite(output_path, annotated_image)
print(f"Imagen con detecciones guardada en: {output_path}")
