import cv2
import numpy as np

# Cargar imagen en escala de grises
imagen = cv2.imread('focos.jpg', cv2.IMREAD_GRAYSCALE)

# Reducir tamaño de la imagen original a la mitad
imagen_reducida = cv2.resize(imagen, (imagen.shape[1] // 2, imagen.shape[0] // 2))

# Aplicar filtro Canny a la imagen reducida
canny = cv2.Canny(imagen_reducida, 100, 200)

# Combinar original y filtrada lado a lado
combinada = np.hstack((imagen_reducida, canny))

# Mostrar la imagen combinada
cv2.imshow('Original y Canny (Reducidas)', combinada)
cv2.waitKey(0)
cv2.destroyAllWindows()
