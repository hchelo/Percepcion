import cv2
import mediapipe as mp

# Inicializa MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Configuración de los estilos de dibujo
drawing_styles = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)  # Azul (BGR), grosor de línea
landmark_styles = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=5)  # Rojo para los puntos clave (opcional)

# Cargar la imagen
image_path = 'dedos/3_4.png'  # Cambia esto con la ruta de tu imagen
image = cv2.imread(image_path)

# Convierte la imagen a RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Inicializa el modelo Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Procesa la imagen para obtener los puntos de pose
    results = pose.process(image_rgb)

    # Verifica si se detectaron puntos clave
    if results.pose_landmarks:
        # Dibuja los puntos de la pose con el color y grosor especificado
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_styles, connection_drawing_spec=drawing_styles)

    else:
        print("No se detectaron personas.")

# Mostrar la imagen con los puntos clave detectados
cv2.imshow("Detección de Persona", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
