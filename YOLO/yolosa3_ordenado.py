import subprocess
import os
from PIL import Image

def get_detected_file(directory, image_name):
    """
    Encuentra la imagen detectada generada por YOLO.
    :param directory: Directorio base de detección (por ejemplo, 'runs/detect').
    :param image_name: Nombre de la imagen original.
    :return: Ruta completa de la imagen detectada.
    """
    base_name, ext = os.path.splitext(image_name)
    detect_dir = os.path.join(directory, f"{base_name}_detected")

    if not os.path.exists(detect_dir):
        raise FileNotFoundError(f"No se encontró la carpeta de detección: {detect_dir}")

    detected_file = os.path.join(detect_dir, image_name)
    if not os.path.exists(detected_file):
        raise FileNotFoundError(f"No se encontró el archivo detectado: {detected_file}")

    return detected_file

# Nombre del archivo de la imagen original
image_filename = 'futbolin.jpg'

# Directorio base donde YOLO guarda las detecciones
output_directory = 'runs/detect'

# Establecer la variable de entorno (opcional según tu configuración)
subprocess.run("set HYDRA_FULL_ERROR=1", shell=True, check=True)

# Ejecutar el comando YOLO, guardando la salida en una carpeta personalizada
subprocess.run(
    f"yolo task=detect mode=predict conf=0.3 model=yolov8n.pt source={image_filename} save_txt=False save_crop=False name=futbolin_detected",
    shell=True, check=True
)

# Buscar el archivo detectado
try:
    detected_file_path = get_detected_file(output_directory, image_filename)
    print(f"Archivo detectado encontrado: {detected_file_path}")

    # Combinar imágenes
    img_original = Image.open(image_filename)
    img_detected = Image.open(detected_file_path)

    # Crear una nueva imagen que combine ambas
    combined_width = img_original.width + img_detected.width
    combined_height = max(img_original.height, img_detected.height)
    combined_img = Image.new('RGB', (combined_width, combined_height))

    # Pegar las imágenes lado a lado
    combined_img.paste(img_original, (0, 0))
    combined_img.paste(img_detected, (img_original.width, 0))

    # Guardar y mostrar la imagen combinada
    base_name, ext = os.path.splitext(image_filename)
    combined_result_filename = f"{base_name}_combined{ext}"
    combined_img.save(combined_result_filename)
    combined_img.show()

    print(f"Imagen combinada guardada como: {combined_result_filename}")

except Exception as e:
    print(f"Error: {e}")
