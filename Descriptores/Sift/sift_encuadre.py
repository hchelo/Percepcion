import cv2
import numpy as np
# Cargar las imágenes y convertirlas a escala de grises
img1 = cv2.imread('fanta.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('fantaB.png')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Crear el objeto SIFT
sift = cv2.SIFT_create()

# Detectar keypoints y calcular los descriptores para ambas imágenes
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Emparejar keypoints utilizando el algoritmo de fuerza bruta
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Aplicar el ratio test para seleccionar los mejores matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Dibujar los mejores matches entre las dos imágenes
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Obtener las coordenadas de los keypoints en ambas imágenes
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Calcular la homografía entre las dos imágenes
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

# Obtener las dimensiones de la imagen 1
h, w = img1.shape[:2]

# Definir los puntos de la zona de interés en la imagen 1
pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

# Proyectar los puntos de la zona de interés de la imagen 1 a la imagen 2 utilizando la homografía
dst = cv2.perspectiveTransform(pts, H)

# Encerrar la zona de interés en un rectángulo en la imagen 2
img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

# Mostrar las dos imágenes con la zona de interés encerrada en un rectángulo
cv2.imshow('Image 1', img1)
cv2.imshow('Image 2', img2)

# Esperar la tecla 'q' para salir del programa
cv2.waitKey(0)
cv2.destroyAllWindows()
