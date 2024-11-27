import cv2

# Cargar la imagen y convertirla a escala de grises
img = cv2.imread('fanta.jpg')
factor_de_escala = 0.1  # Cambia esto según lo pequeño que quieras hacer la imagen, por ejemplo 0.5 es 50%
ancho_nuevo = int(img.shape[1] * factor_de_escala)
alto_nuevo = int(img.shape[0] * factor_de_escala)
img_escalada = cv2.resize(img, (ancho_nuevo, alto_nuevo), interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Crear el objeto SIFT
sift = cv2.SIFT_create()

# Detectar keypoints y calcular los descriptores para la imagen
kp_img, des_img = sift.detectAndCompute(gray, None)

# Crear objeto VideoCapture para cargar el video
cap = cv2.VideoCapture('fanta3.mp4')

while True:
    # Leer un frame del video
    ret, frame = cap.read()
    
    # Convertir a escala de grises
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar keypoints y calcular los descriptores para el frame del video
    kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)

    # Emparejar keypoints utilizando el algoritmo de fuerza bruta
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_img, des_frame, k=2)
    #matches = sorted(matches, key=lambda x:x.distance)

    # Aplicar ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good_matches.append(m)

    # Dibujar los keypoints y emparejamientos en el frame del video
    #img_matches = cv2.drawMatches(img, kp_img, frame, kp_frame, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #print(good_matches)
    img_matches = cv2.drawMatches(img, kp_img, frame, kp_frame, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Mostrar el frame del video con los keypoints y emparejamientos
    #cv2.rectangle(img_matches,(300,80),(450,230),(0,255,0),2)
    cv2.imshow('SIFT', img_matches)
    
    # Esperar la tecla 'q' para salir del bucle while
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y destruir todas las ventanas
cap.release()
cv2.destroyAllWindows()