import cv2

# Cargar imágenes
img1 = cv2.imread('fanta.jpg')
img2 = cv2.imread('fantaB.png')

# Convertir a escala de grises
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Crear objeto SIFT
#sift = cv2.xfeatures2d.SIFT_create()
sift = cv2.SIFT_create()

# Detectar características y descripciones en las dos imágenes
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Configurar el algoritmo KNN
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Aplicar ratio test
good_matches = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good_matches.append(m)

# Dibujar los buenos matches en la imagen
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Mostrar la imagen con los buenos matches
cv2.imshow('Imagen con buenos matches SIFT', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()