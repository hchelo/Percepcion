import cv2
import numpy as np

# Cargar la imagen
#imagen = cv2.imread("focos.jpg")
imagen = cv2.imread("focos.jpg")

# Redimensionar la imagen (por ejemplo, a la mitad de su tamaño)
imagen_reducida = cv2.resize(imagen, (0, 0), fx=0.5, fy=0.5)

# Convertir la imagen reducida a escala de grises
imagen_gris = cv2.cvtColor(imagen_reducida, cv2.COLOR_BGR2GRAY)

# Aplicar el filtro Canny para detectar bordes
bordes = cv2.Canny(imagen_gris,100, 200)

# Aplicar la Transformada de Hough Probabilística para detectar líneas
lineas = cv2.HoughLinesP(bordes, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=100)

# Dibujar las líneas detectadas en la imagen reducida
imagen_filtrada = imagen_reducida.copy()
if lineas is not None:
    for linea in lineas:
        x1, y1, x2, y2 = linea[0]
        cv2.line(imagen_filtrada, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Combinar la original y la filtrada lado a lado
combinada = np.hstack((imagen_reducida, imagen_filtrada))

# Mostrar el resultado
cv2.imshow("Original y Filtrada", combinada)
cv2.waitKey(0)
cv2.destroyAllWindows()
