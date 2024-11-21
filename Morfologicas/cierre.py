import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen en escala de grises
image = cv2.imread("dactilar.jpg", cv2.IMREAD_GRAYSCALE)

# Crear un kernel para la operación de cierre
kernel = np.ones((5, 5), np.uint8)  # Kernel de 5x5 de unos

# Aplicar operación de cierre
closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Mostrar la imagen original y la imagen con cierre
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Imagen con Cierre")
plt.imshow(closed_image, cmap="gray")
plt.axis("off")

plt.show()
