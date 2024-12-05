import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Ruta al modelo
MODEL_PATH = "models/efficientdet_lite0_int8.tflite"
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# Configuración de visualización de bounding boxes
MARGIN = 10  # píxeles
ROW_SIZE = 10  # píxeles
FONT_SIZE = 1
FONT_THICKNESS = 2
TEXT_COLOR = (0, 255, 255)  # amarillo

def visualize(image: np.ndarray, detection_result) -> np.ndarray:
    """Dibuja las cajas delimitadoras y etiquetas en la imagen."""
    for detection in detection_result.detections:
        # Coordenadas de la caja delimitadora
        bbox = detection.bounding_box
        start_point = int(bbox.origin_x), int(bbox.origin_y)
        end_point = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 2)

        # Etiqueta y probabilidad
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f"{category_name} ({probability})"
        text_location = (int(MARGIN + bbox.origin_x), int(MARGIN + ROW_SIZE + bbox.origin_y))
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image

# Captura de video (puedes usar un video o webcam)
video_path = "cocacola.mp4"  # Cambia por 0 para usar la webcam
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

# Obtener el ancho, alto y fps del video original
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Crear VideoWriter para guardar el video con detecciones
output_video_path = "detected_coca.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para .mp4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir imagen BGR a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Crear una imagen de MediaPipe
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Realizar detección
    start_time = time.time()
    detection_result = detector.detect(mp_frame)
    inference_time = (time.time() - start_time) * 1000

    # Visualizar detecciones
    annotated_frame = visualize(frame, detection_result)

    # Mostrar FPS en pantalla
    fps_text = f"Inference Time: {inference_time:.2f} ms"
    cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    # Mostrar resultados en tiempo real
    cv2.imshow("Detecciones en Tiempo Real", annotated_frame)

    # Escribir el fotograma con detecciones en el archivo de salida
    out.write(annotated_frame)

    # Salir si se presiona 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video con detecciones guardado en: {output_video_path}")
