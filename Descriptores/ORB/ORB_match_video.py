import cv2
import os

# Ruta de la imagen y el video
ruta_imagen = 'C:/Users/marce/OneDrive/Documentos/Pythoncito/VA24/Percepcion/Descriptores/test_img/cocacola.jpg'
ruta_video = 'C:/Users/marce/OneDrive/Documentos/Pythoncito/VA24/Percepcion/Descriptores/test_img/cocacola2.mp4'

# Verificar si la imagen y el video existen
if not os.path.exists(ruta_imagen): 
    print(f"Error: La imagen no existe. Verifica la ruta: {ruta_imagen}")
    exit()
if not os.path.exists(ruta_video):
    print(f"Error: El video no existe. Verifica la ruta: {ruta_video}")
    exit()

# Cargar la imagen en color y convertirla a escala de grises
img_color = cv2.imread(ruta_imagen)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Crear el detector ORB con parámetros ajustados
orb = cv2.ORB_create(nfeatures=500)  # Ajusta el número de características

# Detectar y calcular descriptores de la imagen
kp_img, des_img = orb.detectAndCompute(img_gray, None)

# Verificar que haya descriptores válidos
if des_img is None:
    print("No se encontraron descriptores en la imagen.")
    exit()

# Inicializar el objeto VideoCapture
cap = cv2.VideoCapture(ruta_video)

# Leer los fotogramas del video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Fin del video o error al leer el fotograma.")
        break

    # Convertir el fotograma a escala de grises
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar y calcular descriptores del fotograma
    kp_frame, des_frame = orb.detectAndCompute(frame_gray, None)

    # Verificar que haya descriptores válidos en el fotograma
    if des_frame is not None:
        # Emparejar descriptores utilizando el matcher de fuerza bruta
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_img, des_frame)

        # Ordenar los emparejamientos por distancia
        matches = sorted(matches, key=lambda x: x.distance)

        # Umbral para la distancia de los emparejamientos
        umbral_distancia = 30  # Ajusta el umbral según sea necesario
        matches_filtrados = [m for m in matches if m.distance < umbral_distancia]

        # Dibujar los mejores emparejamientos utilizando imágenes en color
        img_matches = cv2.drawMatches(
            img_color, kp_img, frame, kp_frame, matches_filtrados[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Mostrar los emparejamientos
        cv2.imshow("Emparejamientos entre imagen y video", img_matches)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el objeto VideoCapture y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
