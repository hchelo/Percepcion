import cv2
import numpy as np

# Ruta de la imagen
imagen_path = 'Tomates.jpg'  # Actualiza esta línea con la ruta correcta

# Leer la imagen
img = cv2.imread(imagen_path)

# Verificar si la imagen fue cargada correctamente
if img is None:
    print("Error al cargar la imagen.")
else:
    # Reducir el tamaño de la imagen a la mitad
    img_resized = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

    # Crear una copia para aplicar el filtro
    img_filtered = img_resized.copy()

    # Obtener las dimensiones de la imagen redimensionada
    height, width, _ = img_filtered.shape

    # Aplicar el filtro de detección de color
    for y in range(height):
        for x in range(width):
            b, g, r = img_filtered[y, x]
            if (r < 250) and (r >= 154) and (g < 120) and (g >= 0) and (b < 42) and (r >= 2) :
                img_filtered[y, x] = (255, 0,0 )  # Rojo para las áreas detectadas
            else:
                img_filtered[y, x] = (0, 0, 0)  # Negro para el resto

    # Crear una nueva imagen donde se combinen original y filtrada
    combined_image = np.hstack((img_resized, img_filtered))

    # Mostrar la imagen combinada
    cv2.imshow('Imagen Original y Filtrada', combined_image)
    cv2.waitKey(0)  # Espera hasta que se presione una tecla
    cv2.destroyAllWindows()
