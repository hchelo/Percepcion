import cv2
import os

# Rutas de las imágenes
ruta_img1 = 'fanta.jpg'
ruta_img2 = 'fantaB.png'

# Verificar si las imágenes existen
if not os.path.exists(ruta_img1) or not os.path.exists(ruta_img2):
    print(f"Error: Una de las imágenes no existe. Verifica las rutas:\n{ruta_img1}\n{ruta_img2}")
    exit()

# Cargar imágenes en color
img1_color = cv2.imread(ruta_img1)
img2_color = cv2.imread(ruta_img2)

# Convertir las imágenes a escala de grises para la detección de puntos clave
img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

# Crear el detector ORB
orb = cv2.ORB_create(nfeatures=500)

# Detectar y calcular descriptores
kp1, des1 = orb.detectAndCompute(img1_gray, None)
kp2, des2 = orb.detectAndCompute(img2_gray, None)

# Verificar que haya descriptores válidos
if des1 is None or des2 is None:
    print("No se encontraron descriptores en una o ambas imágenes.")
    exit()

# Emparejar descriptores utilizando el matcher de fuerza bruta
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Ordenar los emparejamientos por distancia
matches = sorted(matches, key=lambda x: x.distance)

# Dibujar los mejores emparejamientos utilizando imágenes en color
img_matches = cv2.drawMatches(
    img1_color, kp1, img2_color, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Mostrar los emparejamientos
cv2.imshow("Emparejamientos entre imágenes (a color)", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
