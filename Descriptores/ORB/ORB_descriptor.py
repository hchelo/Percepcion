import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Cargar imagen en color y en escala de grises
img = cv.imread('fanta.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Iniciar el detector ORB
orb = cv.ORB_create()

# Detectar los puntos clave con ORB
kp = orb.detect(gray, None)

# Calcular los descriptores con ORB
kp, des = orb.compute(gray, kp)

# Limitar el número de puntos clave que se mostrarán
num_keypoints = 50  # Ajusta este número según cuántos puntos deseas mostrar
kp = sorted(kp, key=lambda x: -x.response)  # Ordenar puntos clave por su respuesta
kp = kp[:num_keypoints]  # Seleccionar solo los primeros N puntos clave

# Dibujar los puntos clave seleccionados con sus tamaños y direcciones
img_with_kp = cv.drawKeypoints(
    gray, kp, None, color=(0, 255, 0), flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
)

# Mostrar la imagen con los puntos clave seleccionados
plt.imshow(img_with_kp)
plt.show()
