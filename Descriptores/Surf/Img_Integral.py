import cv2
import numpy as np

# Cargar la imagen en escala de grises
img = cv2.imread('C:/Users/marce/OneDrive/Documentos/Pythoncito/VA24/Percepcion/Descriptores/test_img/fanta.jpg', cv2.IMREAD_GRAYSCALE)

# Verificar si la imagen fue cargada correctamente
if img is None:
    print("Error al cargar la imagen.")
    exit()

# Calcular la imagen integral
# La imagen integral es una matriz con valores acumulados de la imagen original
integral_img = cv2.integral(img)

# Convertir la imagen integral a un tipo adecuado para mostrarla
# Convertimos a CV_8U para que sea compatible con cv2.imshow
integral_img_display = integral_img[1:, 1:].astype(np.uint8)

# Mostrar la imagen original y su imagen integral
cv2.imshow('Imagen Original', img)
cv2.imshow('Imagen Integral', integral_img_display)

# Esperar a que se presione una tecla para cerrar
cv2.waitKey(0)
cv2.destroyAllWindows()
