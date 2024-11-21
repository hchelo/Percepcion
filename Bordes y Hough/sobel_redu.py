import cv2
import numpy as np

# Cargar imagen en escala de grises
imagen = cv2.imread('focos.jpg', cv2.IMREAD_GRAYSCALE)

# Reducir tamaño de la imagen original a la mitad
imagen_reducida = cv2.resize(imagen, (imagen.shape[1] // 2, imagen.shape[0] // 2))

# Aplicar filtro Sobel
grad_x = cv2.Sobel(imagen_reducida, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(imagen_reducida, cv2.CV_64F, 0, 1, ksize=3)

# Magnitud del gradiente
sobel = cv2.magnitude(grad_x, grad_y)

# Convertir la magnitud a formato de 8 bits para visualización
sobel_8u = np.uint8(sobel)

# Combinar original y filtrada lado a lado
combinada = np.hstack((imagen_reducida, sobel_8u))

# Mostrar la imagen combinada
cv2.imshow('Original y Sobel (Reducidas)', combinada)
cv2.waitKey(0)
cv2.destroyAllWindows()
