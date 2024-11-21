import cv2
import numpy as np
import matplotlib.pyplot as plt

# Leer la imagen
img = cv2.imread('chino.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Suavizar con un filtro gaussiano
smooth = cv2.GaussianBlur(gray, (95, 95), 0)

# Dividir la imagen en escala de grises por la suavizada
division = cv2.divide(gray, smooth, scale=192)

# Crear una figura para mostrar las imágenes y sus histogramas
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Mostrar la imagen original
axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title('Imagen Original')
axes[0, 0].axis('off')

# Mostrar la imagen filtrada
axes[0, 1].imshow(division, cmap='gray')
axes[0, 1].set_title('Imagen Filtrada')
axes[0, 1].axis('off')

# Calcular y mostrar el histograma de la imagen original
hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
axes[1, 0].plot(hist_gray, color='black')
axes[1, 0].set_title('Histograma Original')
axes[1, 0].set_xlim([0, 256])

# Calcular y mostrar el histograma de la imagen filtrada
hist_division = cv2.calcHist([division], [0], None, [256], [0, 256])
axes[1, 1].plot(hist_division, color='black')
axes[1, 1].set_title('Histograma Filtrado')
axes[1, 1].set_xlim([0, 256])

# Ajustar los espacios entre subplots
plt.tight_layout()

# Mostrar la figura
plt.show()
