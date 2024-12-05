import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Inicializar FaceMesh para la segmentación precisa de la cara
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.2)

# Inicializar Hands para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.2)

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir imagen de BGR a RGB
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen con el modelo de segmentación
    results = segmentation.process(rgb_image)

    # Generar la máscara binaria: 1 para la persona y 0 para el fondo
    mask = (results.segmentation_mask > 0.5).astype(np.uint8)

    # Crear la capa del fondo verde (en formato BGR: verde es (0, 255, 0))
    green_background = np.zeros_like(frame, dtype=np.uint8)
    green_background[mask == 0] = (0, 255, 0)  # Fondo verde

    # Crear una capa de la persona con color azul (en formato BGR: azul es (255, 0, 0))
    blue_person = np.zeros_like(frame, dtype=np.uint8)
    blue_person[mask == 1] = (255, 0, 0)  # Persona azul

    # Detectar los puntos de referencia de la cara
    face_results = face_mesh.process(rgb_image)

    # Crear una capa para la cara de color rojo
    red_face = np.zeros_like(frame, dtype=np.uint8)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Extraer los puntos de referencia de la cara (solo los puntos de la cara, no los del cuerpo)
            face_points = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                face_points.append((x, y))

            # Definir los puntos exteriores del rostro (por ejemplo, la parte externa de los ojos, cejas, nariz, etc.)
            face_points = np.array(face_points, dtype=np.int32)

            # Aplicar Convex Hull para obtener el contorno externo de la cara (sin puntos dilatados)
            hull = cv2.convexHull(face_points)

            # Dibujar el área de la cara de color rojo (Asegúrate de usar BGR (0, 0, 255) para rojo)
            cv2.fillPoly(red_face, [hull], (0, 0, 255))  # Rojo en formato BGR

    # Detectar las manos
    hands_results = hands.process(rgb_image)
    
    # Crear una capa para las manos de color naranja
    orange_hands = np.zeros_like(frame, dtype=np.uint8)

    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            # Extraer los puntos de referencia de la mano
            hand_points = []
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                hand_points.append((x, y))

            # Convertir los puntos de la mano a un array numpy
            hand_points = np.array(hand_points, dtype=np.int32)

            # Dibujar la mano de color naranja
            cv2.fillPoly(orange_hands, [hand_points], (0, 165, 255))  # Naranja en formato BGR

    # Combinar primero el fondo verde y la persona azul
    combined_image = cv2.addWeighted(green_background, 1, blue_person, 1, 0)

    # Luego aplicar la cara roja sobre las capas anteriores
    combined_image = cv2.addWeighted(combined_image, 1, red_face, 1, 0)  # Agregar la cara roja

    # Finalmente, agregar las manos de color naranja
    combined_image = cv2.addWeighted(combined_image, 1, orange_hands, 1, 0)  # Agregar las manos naranjas

    # Mostrar el resultado
    cv2.imshow("Segmentación con Fondo Verde, Persona Azul, Cara Roja y Manos Naranjas", combined_image)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
