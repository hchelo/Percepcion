import subprocess
import os
from datetime import datetime
from PIL import Image
# para videos yolo task=detect mode=predict conf=0.8 model=yolov8n.pt source=jeepepetas.mp4
# Nombre del archivo de la imagen original
image_filename = 'futbolin.jpg'

# Establecer la variable de entorno
subprocess.run("set HYDRA_FULL_ERROR=1", shell=True, check=True)

# Ejecutar el comando YOLO
subprocess.run(f"yolo task=detect mode=predict conf=0.3 model=yolov8n.pt source={image_filename}", shell=True, check=True)

def get_most_recent_file(directory):
    if not os.path.exists(directory):
        print(f"Error: el directorio {directory} no existe.")
        return None

    most_recent_file = None
    most_recent_time = None

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_time = os.path.getmtime(file_path)
            if most_recent_time is None or file_time > most_recent_time:
                most_recent_file = file_path
                most_recent_time = file_time

    return most_recent_file

# Ruta del directorio que deseas leer
directory = 'runs/detect'

# Obtener el archivo más reciente
most_recent_file = get_most_recent_file(directory)

if most_recent_file:
    print(f'El archivo más reciente es: {most_recent_file}')
else:
    print('No se encontraron archivos en el directorio.')

# Si el archivo más reciente es una imagen JPG, combinar y mostrar las imágenes
if most_recent_file and most_recent_file.lower().endswith('.jpg'):
    try:
        img_original = Image.open(image_filename)
        img_processed = Image.open(most_recent_file)

        # Crear una nueva imagen que combine ambas
        combined_width = img_original.width + img_processed.width
        combined_height = max(img_original.height, img_processed.height)

        combined_img = Image.new('RGB', (combined_width, combined_height))

        # Pegar las imágenes lado a lado
        combined_img.paste(img_original, (0, 0))
        combined_img.paste(img_processed, (img_original.width, 0))

        # Guardar la imagen combinada
        combined_img.save("combined_result.jpg")

        # Mostrar la imagen combinada
        combined_img.show()

    except Exception as e:
        print(f'Error al abrir la imagen: {e}')
