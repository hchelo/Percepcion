import cv2
import mediapipe as mp

# Inicializa MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Configuración de los estilos de dibujo
drawing_styles = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)  # Verde (BGR), grosor de línea
landmark_styles = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)  # Rojo para los puntos clave

# Cargar el video
video_path = 'Dragon.mp4'  # Cambia esto con la ruta de tu video
cap = cv2.VideoCapture(video_path)

# Verifica si el video se abre correctamente
if not cap.isOpened():
    print("Error al abrir el video.")
    exit()

# Inicializa el modelo Face Mesh
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        # Lee un fotograma del video
        ret, frame = cap.read()
        
        if not ret:
            break  # Si no hay más fotogramas, termina el bucle

        # Convierte el fotograma a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesa el fotograma para obtener los puntos de la cara
        results = face_mesh.process(frame_rgb)

        # Verifica si se detectaron puntos clave en la cara
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Índices de algunos puntos clave en la cara (ojos, nariz, boca)
                # Ojos (más puntos clave para una mejor visualización)
                left_eye = [33, 133, 160, 159, 158, 157]  # Ojo izquierdo
                right_eye = [362, 263, 249, 390, 373, 374]  # Ojo derecho
                # Nariz (base de la nariz y puntos cercanos)
                nose = [1, 2, 6, 4, 5]  # Puntos de la nariz
                # Boca (bordes y centro)
                mouth = [13, 14, 61, 291]  # Comisuras de la boca y puntos cercanos

                # Dibuja los puntos clave seleccionados
                for idx in left_eye + right_eye + nose + mouth:
                    x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                    y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                    cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)  # Dibuja el punto en rojo

        # Muestra el fotograma con los puntos clave de la cara detectados
        cv2.imshow("Detección de Puntos en la Cara", frame)

        # Si presionas 'q', se detendrá la ejecución del video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Libera el objeto de captura de video y cierra las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()
