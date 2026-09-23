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
        if (r <= 127) and (r >= 110) and (g <= 241) and (g >= 224) and (b <= 242) and (b >= 238):
            img[y, x] = (255, 255, 255) 
        elif (r <= 176) and (r >= 166) and (g <= 240) and (g >= 228) and (b <= 66) and (b >= 45):
            img[y, x] = (255, 255, 255) 
        elif (r <= 255) and (r >= 249) and (g <= 136) and (g >= 103) and (b <= 184) and (b >= 156):
            img[y, x] = (255, 255, 255)     
        else:
            img[y, x] = (0, 0, 0)  # Cambiar el color a negro si no cumple con los criterios

# Mostrar y guardar la imagen resultante
cv2.imshow('Imagen Filtrada', img)
cv2.waitKey(0)  # Mantener la imagen hasta que se presione una tecla
cv2.destroyAllWindows()

# Guardar la imagen filtrada si es necesario
#cv2.imwrite('imagen_filtrada.jpg', img)
