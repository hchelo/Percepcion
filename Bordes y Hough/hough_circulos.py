import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread("focos.jpg")

# Redimensionar la imagen (por ejemplo, a la mitad de su tamaño)
imagen_reducida = cv2.resize(imagen, (0, 0), fx=0.5, fy=0.5)

# Convertir la imagen reducida a escala de grises
imagen_gris = cv2.cvtColor(imagen_reducida, cv2.COLOR_BGR2GRAY)

# Aplicar un desenfoque para reducir el ruido
imagen_gris = cv2.GaussianBlur(imagen_gris, (9, 9), 2)

# Detectar círculos con la Transformada de Hough
circulos = cv2.HoughCircles(
    imagen_gris,
    cv2.HOUGH_GRADIENT,
    dp=1,  # Inverso de la resolución del acumulador
    minDist=20,  # Distancia mínima entre los centros de los círculos detectados
    param1=50,  # Umbral para el detector de bordes (Canny)
    param2=18,  # Umbral para el acumulador (detección de círculos)
    minRadius=10,  # Radio mínimo del círculo
    maxRadius=50   # Radio máximo del círculo
)

# Dibujar los círculos detectados en la imagen reducida
imagen_circulos = imagen_reducida.copy()
if circulos is not None:
    circulos = np.uint16(np.around(circulos))  # Redondear valores al entero más cercano
    for circulo in circulos[0, :]:
        # Dibujar el borde del círculo
        cv2.circle(imagen_circulos, (circulo[0], circulo[1]), circulo[2], (0, 255, 0), 2)
        # Dibujar el centro del círculo
        cv2.circle(imagen_circulos, (circulo[0], circulo[1]), 2, (0, 0, 255), 3)

# Combinar la original y la filtrada lado a lado
combinada = np.hstack((imagen_reducida, imagen_circulos))

# Mostrar el resultado
cv2.imshow("Original y con Círculos Detectados", combinada)
cv2.waitKey(0)
cv2.destroyAllWindows()
