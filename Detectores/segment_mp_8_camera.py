import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageSequence

# Inicializar MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Cargar el GIF animado
gif_path = "baseValue.gif"
gif = Image.open(gif_path)
gif_frames = [np.array(frame.convert("RGB")) for frame in ImageSequence.Iterator(gif)]

# Captura de video (puedes usar un video o webcam)
cap = cv2.VideoCapture(0)

# Inicializar índice de fotogramas del GIF
frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir imagen BGR a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar imagen
    results = segmentation.process(rgb_frame)

    # Generar la máscara
    mask = results.segmentation_mask > 0.5  # Umbral para decidir qué es "primer plano"

    # Obtener el fotograma actual del GIF y redimensionarlo
    gif_frame = gif_frames[frame_index % len(gif_frames)]
    gif_frame = cv2.resize(gif_frame, (frame.shape[1], frame.shape[0]))

    # Convertir GIF a formato BGR (OpenCV usa BGR)
    gif_frame_bgr = cv2.cvtColor(gif_frame, cv2.COLOR_RGB2BGR)

    # Combinar el primer plano con el fondo del GIF
    output_frame = np.where(mask[..., None], frame, gif_frame_bgr)

    # Mostrar resultados
    cv2.imshow("Segmentacion con Fondo Animado (GIF)", output_frame)

    # Avanzar al siguiente fotograma del GIF
    frame_index += 1

    if cv2.waitKey(30) & 0xFF == 27:  # Presiona 'ESC' para salir
        break

cap.release()
cv2.destroyAllWindows()
