import cv2
import mediapipe as mp

# Inicializa MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Configuración de los estilos de dibujo
drawing_styles = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)  # Verde (BGR), grosor de línea
landmark_styles = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)  # Malla de la cara en verde

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
                # Dibuja los contornos de la cara (sin puntos)
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None, connection_drawing_spec=drawing_styles)

        # Muestra el fotograma con la malla de la cara detectada
        cv2.imshow("Malla de la Cara", frame)

        # Si presionas 'q', se detendrá la ejecución del video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Libera el objeto de captura de video y cierra las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()
