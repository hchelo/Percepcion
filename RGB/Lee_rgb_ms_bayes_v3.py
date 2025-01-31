import cv2
import numpy as np

# Ruta completa de la imagen
imagen_path = 'Tomates.jpg'  # Actualiza esta línea con la ruta correcta

# Leer la imagen
imagen = cv2.imread(imagen_path)

# Convertir la imagen a formato flotante para cálculos precisos
imagen_float = np.float64(imagen)

# Leer los valores de media, desviación estándar y matriz de covarianza desde el archivo 'valores_rgb_unicos.txt'
valores_rgb = []

with open('valores_rgb_unicos.txt', 'r') as f:
    for line in f:
        rgb = tuple(map(int, line.strip().split(', ')))
        valores_rgb.append(rgb)

# Convertir la lista de valores RGB en una matriz numpy
MatrizRGB = np.array(valores_rgb)

# Calcular la media de cada canal (R, G, B)
media_1 = np.mean(MatrizRGB, axis=0)

# Calcular la matriz de covarianza
covarianza_1 = np.cov(MatrizRGB, rowvar=False)

# Crear la matriz de covarianza diagonalizada
cova_1 = np.diag(np.diag(covarianza_1))

# Calcular el determinante de la matriz de covarianza
deter_1 = np.linalg.det(cova_1)

# Calcular la inversa de la matriz de covarianza
inversa_1 = np.linalg.inv(cova_1)

# Inicializar una nueva imagen para almacenar los píxeles filtrados
new_RGB = np.zeros_like(imagen, dtype=np.uint8)

# Pre-calcular la constante de normalización para Pk_1
constante_Pk_1 = 1 / np.sqrt(((2 * np.pi) ** 3) * deter_1)

# Vectorizar el cálculo de la distancia de Mahalanobis para todos los píxeles
imagen_rgb_float = imagen_float.reshape((-1, 3))  # Convertir la imagen en una lista de píxeles RGB

# Calcular el Sk_1 para todos los píxeles de la imagen
dif_pixel_media = imagen_rgb_float - media_1  # Diferencia entre cada píxel y la media
Sk_1 = np.sum(dif_pixel_media @ inversa_1 * dif_pixel_media, axis=1)  # Cálculo vectorizado de Sk_1

# Calcular Pk_1 para todos los píxeles
Pk_1 = constante_Pk_1 * np.exp(-Sk_1 / 2)
umbral=0.00000005
# Filtrar los píxeles según el valor de Pk_1
new_RGB_flat = np.zeros_like(imagen_rgb_float, dtype=np.uint8)
new_RGB_flat[Pk_1 > umbral] = [0, 0, 255]  # Color rojo
new_RGB_flat[Pk_1 <= umbral] = [0, 0, 0]   # Color negro

# Reconstruir la imagen filtrada desde la lista de píxeles
new_RGB = new_RGB_flat.reshape(imagen.shape)

# Mostrar la imagen original
cv2.imshow('Imagen Original', imagen)

# Mostrar la nueva imagen filtrada
cv2.imshow('Imagen Filtrada', new_RGB)

# Esperar hasta que se presione una tecla para cerrar
cv2.waitKey(0)
cv2.destroyAllWindows()
