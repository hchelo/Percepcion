import cv2
import numpy as np
import matplotlib.pyplot as plt

# Leer la imagen en escala de grises
im = cv2.imread("blobs3.jpg", cv2.IMREAD_GRAYSCALE)

# Umbralizar la imagen para que los objetos blancos sean separados del fondo negro
_, thresholded_image = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)

# Detectar contornos de los objetos blancos
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Crear una copia de la imagen para dibujar los contornos
im_with_contours = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # Convertir a BGR para dibujar

# Dibujar los contornos detectados
cv2.drawContours(im_with_contours, contours, -1, (0, 0, 255), 2)  # Rojo para los contornos

# Agregar una cruz en el centro de cada objeto
for contour in contours:
    # Calcular el centro de cada objeto usando los momentos
    M = cv2.moments(contour)
    if M["m00"] != 0:  # Evitar la división por cero
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Dibujar una cruz en el centro del objeto
        size = 10  # Tamaño de la cruz
        cv2.line(im_with_contours, (cX - size, cY), (cX + size, cY), (0, 0, 255), 2)  # Línea horizontal
        cv2.line(im_with_contours, (cX, cY - size), (cX, cY + size), (0, 0, 255), 2)  # Línea vertical

# Crear una figura con Matplotlib para mostrar las imágenes lado a lado
plt.figure(figsize=(12, 6))

# Imagen original
plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(im, cmap="gray")
plt.axis("off")

# Imagen con contornos y cruces
plt.subplot(1, 2, 2)
plt.title("Objetos Blancos Detectados con Contornos y Cruces")
plt.imshow(cv2.cvtColor(im_with_contours, cv2.COLOR_BGR2RGB))  # Convertir a RGB para Matplotlib
plt.axis("off")

plt.tight_layout()
plt.show()
