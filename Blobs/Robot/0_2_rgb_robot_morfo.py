import cv2
import numpy as np

# Ruta al archivo de imagen
archivo_imagen = 'soccer3.jpg'  # Actualiza esta línea con la ruta correcta del archivo de imagen

# Leer la imagen
img = cv2.imread(archivo_imagen)
if img is None:
    raise ValueError(f"No se pudo cargar la imagen desde {archivo_imagen}")

height, width, _ = img.shape

# Aplicar el filtro de detección de piel
for y in range(height):
    for x in range(width):
        b, g, r = img[y, x]
        if (r <= 180) and (r >= 96) and (g <= 251) and (g >= 204) and (b <= 255) and (b >= 207):
            img[y, x] = (255, 255, 255) 
        elif (r <= 226) and (r >= 151) and (g <= 255) and (g >= 209) and (b <= 130) and (b >= 44):
            img[y, x] = (255, 255, 255) 
        elif (r <= 255) and (r >= 234) and (g <= 185) and (g >= 76) and (b <= 221) and (b >= 151):
            img[y, x] = (255, 255, 255)     
        else:
            img[y, x] = (0, 0, 0)  # Cambiar el color a negro si no cumple con los criterios

kernel = np.ones((3, 3), np.uint8)  # Kernel de 5x5
# Realizar la erosión
erosion = cv2.erode(img, kernel, iterations=1)
# Realizar la dilatación
kernel = np.ones((50, 50), np.uint8)  # Kernel de 5x5
dilatation = cv2.dilate(erosion, kernel, iterations=1)
img=dilatation

# Mostrar y guardar la imagen resultante
cv2.imshow('Imagen Filtrada', img)
cv2.waitKey(0)  # Mantener la imagen hasta que se presione una tecla
cv2.destroyAllWindows()

# Guardar la imagen filtrada si es necesario
cv2.imwrite('imagen_filtrada.jpg', img)
