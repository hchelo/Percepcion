import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen en escala de grises
image = cv2.imread("hobre.jpg", cv2.IMREAD_GRAYSCALE)

# Crear un kernel para la erosión
kernel = np.ones((3, 3), np.uint8)  # Kernel de 5x5 de unos
"""
kernel = np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]], dtype=np.uint8)
print(kernel)
"""
# Aplicar erosión
eroded_image = cv2.erode(image, kernel, iterations=1)

# Mostrar la imagen original y la imagen erosionada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Imagen Erosionada")
plt.imshow(eroded_image, cmap="gray")
plt.axis("off")

plt.show()
