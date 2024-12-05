import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import os

# Ruta al modelo existente
MODEL_PATH = "models/gesture_recognizer.task"

# Verificar que el modelo exista
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"El modelo no se encontró en la ruta: {MODEL_PATH}")

# Crear el objeto GestureRecognizer
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Cargar la imagen que quieres procesar
image_path = 'gestos/person_victory.png'  # Reemplaza con la ruta de tu imagen
image = cv2.imread(image_path)

if image is None:
    raise ValueError(f"No se pudo cargar la imagen desde {image_path}. Verifica la ruta.")

# Convertir la imagen a RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Crear un objeto mp.Image con el formato correcto
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

# Reconocer los gestos en la imagen de entrada
recognition_result = recognizer.recognize(mp_image)

# Procesar el resultado
if recognition_result.gestures:
    top_gesture = recognition_result.gestures[0][0]
    print(f"Gesto Detectado: {top_gesture.category_name} con una probabilidad de {top_gesture.score:.2f}")
else:
    print("No se detectaron gestos.")

# Dibujar los puntos de referencia y un rectángulo alrededor de la mano
if recognition_result.hand_landmarks:
    for hand_landmarks in recognition_result.hand_landmarks:
        min_x = min([landmark.x for landmark in hand_landmarks])
        max_x = max([landmark.x for landmark in hand_landmarks])
        min_y = min([landmark.y for landmark in hand_landmarks])
        max_y = max([landmark.y for landmark in hand_landmarks])

        # Convertir coordenadas normalizadas a píxeles
        min_x_px = int(min_x * image.shape[1])
        max_x_px = int(max_x * image.shape[1])
        min_y_px = int(min_y * image.shape[0])
        max_y_px = int(max_y * image.shape[0])

        # Dibujar el rectángulo
        cv2.rectangle(image, (min_x_px, min_y_px), (max_x_px, max_y_px), (255, 0, 0), 2)

        # Etiqueta con la probabilidad
        label = f"{top_gesture.category_name}: {top_gesture.score:.2f}"
        cv2.putText(image, label, (min_x_px, min_y_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Mostrar la imagen
cv2.imshow('Gestos y Puntos de Referencia', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
