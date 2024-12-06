from mtcnn import MTCNN
import cv2
import os
import time

# Configuración de las carpetas
folders = {
    "hombres": "feret_test/hombres",  # Cambia a la ruta de la carpeta de hombres
    "mujeres": "feret_test/mujeres"   # Cambia a la ruta de la carpeta de mujeres
}

# Inicializar el detector MTCNN
detector = MTCNN()

# Procesar cada carpeta
for category, folder_path in folders.items():
    print(f"Procesando imágenes en la carpeta: {category}")
    # Obtener todas las imágenes en la carpeta
    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)
        if not os.path.isfile(image_path):
            continue
        
        # Cargar y procesar la imagen
        image = cv2.imread(image_path)
        if image is None:
            print(f"No se pudo leer la imagen: {image_path}")
            continue
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb_image)

        # Dibujar rectángulos para cada cara detectada
        for result in results:
            x, y, width, height = result['box']
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Mostrar la imagen con las detecciones
        cv2.imshow(f"{category}: {file_name}", image)
        cv2.waitKey(300)  # Mostrar la imagen durante 1 segundo
        cv2.destroyAllWindows()

    print(f"Finalizado procesamiento de la carpeta: {category}")

cv2.destroyAllWindows()
