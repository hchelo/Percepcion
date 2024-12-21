import subprocess

# Establecer la variable de entorno
subprocess.run("set HYDRA_FULL_ERROR=1", shell=True, check=True)

# Ejecutar el comando YOLO
subprocess.run("yolo task=detect mode=predict model=yolov8n.pt source=futbolin.jpg", shell=True, check=True)

import os
from datetime import datetime
from PIL import Image

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

# Si el archivo más reciente es una imagen JPG, mostrarla
if most_recent_file and most_recent_file.lower().endswith('.jpg'):
    try:
        img = Image.open(most_recent_file)
        img.show()
    except Exception as e:
        print(f'Error al abrir la imagen: {e}')
