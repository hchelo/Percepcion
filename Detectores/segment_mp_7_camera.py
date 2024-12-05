import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Captura de video (puedes usar un video o webcam)
cap = cv2.VideoCapture(0)

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

    # Crear el fondo desenfocado
    blurred_background = cv2.GaussianBlur(frame, (61, 61), 0)

    # Combinar el primer plano con el fondo desenfocado
    output_frame = np.where(mask[..., None], frame, blurred_background)

    # Mostrar resultados
    cv2.imshow("Segmentación con Fondo Desenfocado", output_frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Presiona 'ESC' para salir
        break

cap.release()
cv2.destroyAllWindows()
