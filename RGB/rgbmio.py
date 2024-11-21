import cv2
import numpy as np
import os

# Directorio donde están las imágenes
directorio_imagenes = 'H:\DATOS IMPORTANTES\Desktop\Infografia\'  # Actualiza esta línea para usar el directorio correcto

# Lista de nombres de archivos de imágenes en el directorio
nombres_imagenes = [img for img in os.listdir(directorio_imagenes) if img.endswith('.jpg')]

# Procesar cada imagen en el directorio
for nombre_imagen in nombres_imagenes:
    path_imagen = os.path.join(directorio_imagenes, nombre_imagen)
    img = cv2.imread(path_imagen)
    height, width, _ = img.shape

    # Aplicar el filtro de detección de piel
    for y in range(height):
        for x in range(width):
            b, g, r = img[y, x]
            if (r > 95) and (g > 40) and (b > 20) and (max(r, g, b) - min(r, g, b) > 15) and (abs(r - g) > 15) and (r > g) and (r > b):
                continue  # Mantener el color original si cumple con los criterios de piel
            else:
                img[y, x] = (0, 0, 0)  # Cambiar el color a negro si no cumple con los criterios

    # Mostrar y guardar la imagen resultante
    cv2.imshow(f'Imagen Filtrada - {nombre_imagen}', img)
    cv2.waitKey(0)  # Cambiado de 500 a 0 para que la imagen permanezca hasta que se presione una tecla
    cv2.destroyAllWindows()

