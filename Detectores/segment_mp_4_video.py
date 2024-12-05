import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Cargar el video
cap = cv2.VideoCapture("Dragon.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir imagen de BGR a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen
    results = segmentation.process(rgb_frame)

    # Generar la máscara binaria
    mask = (results.segmentation_mask > 0.5).astype(np.uint8)

    # Crear un fondo rojo
    red_background = np.zeros_like(frame, dtype=np.uint8)
    red_background[:] = (0, 0, 255)  # Rojo en formato BGR

    # Combinar el primer plano (personas) con el fondo rojo
    output_frame = np.where(mask[..., None] == 1, frame, red_background)

    # Mostrar el resultado
    cv2.imshow("Personas con Fondo Rojo", output_frame)

    # Salir si se presiona la tecla 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
