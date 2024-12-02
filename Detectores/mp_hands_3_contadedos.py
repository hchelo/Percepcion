import cv2
import mediapipe as mp

# Inicializa los módulos de MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Función para contar dedos extendidos
def count_fingers(landmarks, handedness):
    fingers = [False] * 5  # Lista para marcar qué dedos están extendidos
    # Índices de los puntos clave en MediaPipe para cada dedo
    finger_tips = [4, 8, 12, 16, 20]  # Pulgar, índice, medio, anular, meñique
    finger_bottoms = [3, 6, 10, 14, 18]

    # Pulgar: depende de su orientación
    if handedness == "Right":
        fingers[0] = landmarks[finger_tips[0]].x < landmarks[finger_bottoms[0]].x
    else:  # Mano izquierda
        fingers[0] = landmarks[finger_tips[0]].x > landmarks[finger_bottoms[0]].x

    # Otros dedos: verificamos si la punta está por encima del nudillo correspondiente
    for i in range(1, 5):
        fingers[i] = landmarks[finger_tips[i]].y < landmarks[finger_bottoms[i]].y

    return sum(fingers)  # Retorna el número de dedos extendidos

# Ruta de la imagen
image_path = 'dedos/5_8.png'  # Reemplaza con la ruta de tu imagen

# Cargar la imagen
image = cv2.imread(image_path)

# Verifica si la imagen se cargó correctamente
if image is None:
    print("Error al cargar la imagen")
else:
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        # Convertir la imagen a RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Procesar la imagen
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Dibuja la mano
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Contar dedos
                num_fingers = count_fingers(hand_landmarks.landmark, handedness.classification[0].label)

                # Muestra el conteo en la imagen
                label = f"{handedness.classification[0].label} hand: {num_fingers} fingers"
                cv2.putText(image, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar la imagen con las marcas
        cv2.imshow('Finger Counter', image)

        # Espera hasta que se presione cualquier tecla
        cv2.waitKey(0)
        cv2.destroyAllWindows()
