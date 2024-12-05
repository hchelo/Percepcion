import sys
import cv2
import mediapipe as mp
# Imprimir la versión de Python
print("Versión de Python:", sys.version)
# Imprimir la versión de OpenCV
print("Versión de OpenCV:", cv2.__version__)
# Imprimir la versión de MediaPipe
print("Versión de MediaPipe:", mp.__version__)

# Inicializa MediaPipe Hands y Drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Carga el archivo de video
cap = cv2.VideoCapture('Dragon.mp4')  # Usa el nombre del archivo de video

# Configuración de MediaPipe Hands
with mp_hands.Hands(
    static_image_mode=False,  # Procesamiento en tiempo real
    max_num_hands=2,         # Número máximo de manos a detectar
    min_detection_confidence=0.5,  # Confianza mínima de detección
    min_tracking_confidence=0.5    # Confianza mínima de seguimiento
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error al leer el video o fin del video.")
            break

        # Convierte la imagen a RGB (MediaPipe usa RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesa el cuadro con MediaPipe Hands
        results = hands.process(frame_rgb)

        # Dibuja las marcas en la imagen original
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Muestra el cuadro procesado
        cv2.imshow("Hand Detection", frame)

        # Rompe el bucle con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Libera el video y cierra las ventanas
cap.release()
cv2.destroyAllWindows()
