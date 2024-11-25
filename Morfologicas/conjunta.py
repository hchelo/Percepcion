import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen en escala de grises
image = cv2.imread("img_morfo/dactilar.jpg", cv2.IMREAD_GRAYSCALE)

# Binarización de la imagen
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Crear los kernels en forma de cruz
kernel_cross_3x3 = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=np.uint8)  # 3x3 en cruz

kernel_cross_7x7 = np.array([[0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 1],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0]], dtype=np.uint8)  # 7x7 en cruz

# 1. Cierre con kernel 7x7 en cruz
closed_image_1 = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_cross_7x7)

# 2. Erosión con kernel 3x3 en cruz
eroded_image_1 = cv2.erode(closed_image_1, kernel_cross_3x3, iterations=1)

# 3. Cierre con kernel 3x3 en cruz
closed_image_2 = cv2.morphologyEx(eroded_image_1, cv2.MORPH_CLOSE, kernel_cross_3x3)

# 4. Erosión con kernel 3x3 en cruz
eroded_image_2 = cv2.erode(closed_image_2, kernel_cross_3x3, iterations=1)

# 5. Dilatación con kernel 7x7 en cruz
dilated_image = cv2.dilate(eroded_image_2, kernel_cross_7x7, iterations=1)

# Mostrar solo la primera y la última imagen
plt.figure(figsize=(10, 5))

# Imagen original binarizada
plt.subplot(1, 2, 1)
plt.title("Imagen Original Binaria")
plt.imshow(binary_image, cmap="gray")
plt.axis("off")

# Imagen tras la dilatación (7x7)
plt.subplot(1, 2, 2)
plt.title("Dilatación 7x7")
plt.imshow(dilated_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
