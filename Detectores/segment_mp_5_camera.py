import cv2
import mediapipe as mp

# Inicializar MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Leer imagen personalizada de fondo
background_image = cv2.imread("mapa.jpg")

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
    mask = results.segmentation_mask
    condition = mask > 0.5  # Umbral para decidir qué píxeles son "primer plano"

    # Redimensionar fondo a las dimensiones del frame
    background_resized = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))

    # Combinar la imagen de entrada con el fondo personalizado
    output_frame = frame.copy()
    output_frame[~condition] = background_resized[~condition]

    # Mostrar resultados
    cv2.imshow("Segmentacion con Fondo Personalizado", output_frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Presiona 'ESC' para salir
        break

cap.release()
cv2.destroyAllWindows()
