import cv2
import mediapipe as mp

# Inicializar MediaPipe Selfie Segmentation con ambos modelos
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation_0 = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
segmentation_1 = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

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

    # Procesar imagen con ambos modelos
    results_0 = segmentation_0.process(rgb_frame)
    results_1 = segmentation_1.process(rgb_frame)

    # Generar las máscaras
    mask_0 = results_0.segmentation_mask > 0.5
    mask_1 = results_1.segmentation_mask > 0.5

    # Redimensionar fondo a las dimensiones del frame
    background_resized = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))

    # Combinar la imagen de entrada con el fondo personalizado para ambos modelos
    output_0 = frame.copy()
    output_0[~mask_0] = background_resized[~mask_0]

    output_1 = frame.copy()
    output_1[~mask_1] = background_resized[~mask_1]

    # Concatenar las imágenes para comparación lado a lado
    comparison_frame = cv2.hconcat([output_0, output_1])

    # Mostrar resultados
    cv2.imshow("Comparación Modelos: Izq(0) vs Der(1)", comparison_frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Presiona 'ESC' para salir
        break

cap.release()
cv2.destroyAllWindows()
