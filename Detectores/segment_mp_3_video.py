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

    # Crear una capa transparente con el color deseado (en este caso, azul)
    color = (255, 0, 0)  # Azul en formato BGR
    transparent_layer = np.zeros_like(frame, dtype=np.uint8)
    transparent_layer[mask == 1] = color  # Aplicar color solo en la máscara

    # Mezclar la capa transparente con la imagen original
    alpha = 0.5  # Nivel de transparencia (0 = completamente transparente, 1 = completamente opaco)
    highlighted_frame = cv2.addWeighted(frame, 1 - alpha, transparent_layer, alpha, 0)

    # Mostrar el resultado
    cv2.imshow("Objetos Segmentados con Máscara Transparente", highlighted_frame)

    # Salir si se presiona la tecla 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
