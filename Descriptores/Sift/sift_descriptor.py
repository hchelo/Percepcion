import cv2

# Cargar imagen
img = cv2.imread("C:/Users/marce/OneDrive/Documentos/Pythoncito/VA24/Percepcion/Descriptores/test_img/fanta.jpg")

# Convertir a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Crear objeto SIFT
sift = cv2.SIFT_create()

# Detectar características y descripciones en la imagen
kp, des = sift.detectAndCompute(gray, None)

# Dibujar los puntos clave con círculos y direcciones
img_with_kp = cv2.drawKeypoints(
    gray, kp, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
)

# Mostrar la imagen con los puntos clave y direcciones
cv2.imshow('Imagen con puntos clave y direcciones SIFT', img_with_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Opcional: Imprimir los descriptores para verificarlos
if des is not None:
    print("Descriptores encontrados:", des.shape)
else:
    print("No se encontraron descriptores.")
