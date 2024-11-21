import cv2
import os

# Ruta completa de la imagen
imagen_path = 'Tomates.jpg'  # Actualiza esta línea con la ruta correcta

# Leer la imagen
img = cv2.imread(imagen_path)

# Verificar si la imagen fue cargada correctamente
if img is None:
    print("Error al cargar la imagen.")
else:
    height, width, _ = img.shape

    # Aplicar el filtro de detección de piel
    for y in range(height):
        for x in range(width):
            b, g, r = img[y, x]
            if (r > 95) and (g > 40)  and (b > 20) and ((max(max(g,b),r)-min(min(g,b),r)) > 15) and (abs(r-g)>15) and (r>g) and (r>b):
                img[y, x] = (0, 0, 255)
            else:
                img[y, x] = (0, 0, 0)  # Cambiar el color a negro si no cumple con los criterios

    # Mostrar la imagen resultante
    cv2.imshow('Imagen Filtrada - tomate.jpg', img)
    cv2.waitKey(0)  # Espera hasta que se presione una tecla
    cv2.destroyAllWindows()
