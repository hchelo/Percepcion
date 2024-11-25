import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen en escala de grises
image = cv2.imread("img_morfo/dactilar.jpg", cv2.IMREAD_GRAYSCALE)

# Crear un kernel para la apertura
kernel = np.ones((6, 6), np.uint8)  # Kernel de 5x5 de unos

# Aplicar operación de apertura
opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)



# Mostrar la imagen original y la imagen con apertura
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Imagen con Apertura")
plt.imshow(opened_image, cmap="gray")
plt.axis("off")

plt.show()
