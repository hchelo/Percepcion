import os
import cv2
from mtcnn import MTCNN

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

# Inicializar el detector de rostros MTCNN
detector = MTCNN()

# Función para detectar y dibujar rostros en la imagen
def detect_faces(image):
    # Convertir la imagen a RGB antes de pasarla al detector
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detectar rostros en la imagen
    results = detector.detect_faces(rgb_image)
    
    for result in results:
        # Obtener las coordenadas del rostro
        x, y, width, height = result['box']
        # Dibujar un rectángulo amarillo alrededor del rostro
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 255), 4)  # Amarillo en BGR
    return image

# Mostrar las imágenes una por una con detección de rostros
for image in all_images:
    # Detectar rostros
    image_with_faces = detect_faces(image)

    # Mostrar la imagen con rostros detectados
    cv2.imshow("Imagen con Rostros Detectados", image_with_faces)
    
    # Esperar 2 segundos antes de pasar a la siguiente imagen
    cv2.waitKey(50)  # 2000 milisegundos = 2 segundos

# Cerrar todas las ventanas de imágenes después de mostrar todas
cv2.destroyAllWindows()
