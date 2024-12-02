import cv2

def draw_found_faces(detected, image, color: tuple):
    for (x, y, width, height) in detected:
        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            color,
            thickness=2
        )

# Cargar la imagen
image_path = "face_nor/bradd.png"  # Reemplaza con la ruta a tu imagen
image = cv2.imread(image_path)

# Verificar si la imagen se cargó correctamente
if image is None:
    print("Error al cargar la imagen.")
    exit()

# Crear los objetos de los clasificadores en cascada
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

# Convertir la imagen a escala de grises
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detectar todas las caras en la imagen
detected_faces = face_cascade.detectMultiScale(image=grayscale_image, scaleFactor=1.3, minNeighbors=4)
detected_eyes = eye_cascade.detectMultiScale(image=grayscale_image, scaleFactor=1.3, minNeighbors=4)

# Dibujar rectángulos alrededor de las caras y ojos
draw_found_faces(detected_faces, image, (0, 0, 255))  # Rojo para las caras
draw_found_faces(detected_eyes, image, (0, 255, 0))   # Verde para los ojos

# Mostrar la imagen con las detecciones
cv2.imshow('Detección de Rostros y Ojos', image)

# Esperar hasta que se presione una tecla
cv2.waitKey(0)

# Cerrar las ventanas de OpenCV
cv2.destroyAllWindows()
