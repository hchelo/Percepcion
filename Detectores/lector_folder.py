import os
import cv2

# Ruta de la carpeta donde están las imágenes
folder_path = 'hands'  # Cambia esto con la ruta correcta de tu carpeta

# Listar todos los archivos en la carpeta
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Iterar sobre cada archivo de imagen
for image_file in image_files:
    # Obtener la ruta completa de la imagen
    image_path = os.path.join(folder_path, image_file)

    # Leer la imagen
    image = cv2.imread(image_path)

    # Verificar si la imagen se cargó correctamente
    if image is not None:
        # Mostrar la imagen en una ventana
        cv2.imshow(f"Imagen: {image_file}", image)

        # Esperar hasta que el usuario presione una tecla para mostrar la siguiente imagen
        cv2.waitKey(0)

    else:
        print(f"No se pudo leer la imagen {image_file}")

# Cerrar todas las ventanas al finalizar
cv2.destroyAllWindows()
