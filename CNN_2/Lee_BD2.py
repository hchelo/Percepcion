import os
import cv2

# Ruta a la carpeta de fotos docentes
dataset_dir = 'C:/FotosDocentes/'

# Directorios de Hombres y Mujeres
hombres_dir = os.path.join(dataset_dir, 'Hombres')
mujeres_dir = os.path.join(dataset_dir, 'Mujeres')

# Función para cargar y procesar todas las imágenes de una carpeta
def load_and_process_images_from_directory(directory, target_width=600):
    images = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        # Asegurarse de que el archivo sea una imagen
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(img_path)
            if img is not None:
                # Redimensionar la imagen manteniendo la relación de aspecto
                height, width = img.shape[:2]
                aspect_ratio = height / width
                target_height = int(target_width * aspect_ratio)
                resized_image = cv2.resize(img, (target_width, target_height))

                # Convertir a escala de grises
                gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                images.append(gray_image)
            else:
                print(f"Error al cargar la imagen: {img_path}")
    return images

# Cargar y procesar imágenes de ambas carpetas
hombres_images = load_and_process_images_from_directory(hombres_dir)
mujeres_images = load_and_process_images_from_directory(mujeres_dir)

# Combinar las imágenes de ambas carpetas
all_images = hombres_images + mujeres_images

# Mostrar las imágenes una por una
for image in all_images:
    cv2.imshow("Imagen en Escala de Grises", image)
    # Esperar 2 segundos antes de pasar a la siguiente imagen
    cv2.waitKey(500)  # 2000 milisegundos = 2 segundos

# Cerrar todas las ventanas de imágenes después de mostrar todas
cv2.destroyAllWindows()
