import cv2
import mediapipe as mp

# Inicializa MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Configuración de los estilos de dibujo
drawing_styles = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)  # Azul (BGR), grosor de línea
landmark_styles = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=5)  # Rojo para los puntos clave (opcional)

# Cargar el video
video_path = 'Dragon.mp4'  # Cambia esto con la ruta de tu video
cap = cv2.VideoCapture(video_path)

# Verifica si el video se abre correctamente
if not cap.isOpened():
    print("Error al abrir el video.")
    exit()

# Inicializa el modelo Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        # Lee un fotograma del video
        ret, frame = cap.read()
        
        if not ret:
            break  # Si no hay más fotogramas, termina el bucle

        # Convierte el fotograma a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesa el fotograma para obtener los puntos de pose
        results = pose.process(frame_rgb)

        # Verifica si se detectaron puntos clave
        if results.pose_landmarks:
            # Dibuja los puntos de la pose con el color y grosor especificado
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_styles, connection_drawing_spec=drawing_styles)

        # Muestra el fotograma con los puntos clave detectados
        cv2.imshow("Detección de Persona", frame)

        # Si presionas 'q', se detendrá la ejecución del video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Libera el objeto de captura de video y cierra las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()
