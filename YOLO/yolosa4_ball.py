import subprocess
from PIL import Image
import os

# Nombre del archivo de la imagen original
image_filename = 'futbolin.jpg'

# Nombre del archivo detectado
detected_filename = 'futbolin_detected.jpg'

# Establecer la variable de entorno (opcional según tu configuración)
subprocess.run("set HYDRA_FULL_ERROR=1", shell=True, check=True)

# Ejecutar el comando YOLO, asegurando que se guarden los archivos de texto (.txt) con las predicciones
subprocess.run(
    f"yolo task=detect mode=predict conf=0.3 model=yolov8n.pt source={image_filename} save=True save_txt=True save_crop=False name=futbolin_detected",
    shell=True, check=True
)

# Ruta base del directorio de salida
base_output_directory = os.path.join('runs', 'detect')

# Encontrar el directorio más reciente que contiene "futbolin_detected"
output_directory = None
if os.path.exists(base_output_directory):
    subdirs = [d for d in os.listdir(base_output_directory) if os.path.isdir(os.path.join(base_output_directory, d))]
    detected_dirs = [d for d in subdirs if d.startswith('futbolin_detected')]
    if detected_dirs:
        output_directory = os.path.join(base_output_directory, sorted(detected_dirs)[-1])  # Tomar el más reciente

if output_directory:
    # Verificar que la detección haya sido guardada en la ruta esperada
    detected_path = os.path.join(output_directory, os.path.basename(image_filename))

    # Asegurarse de que el archivo detectado haya sido creado
    if os.path.exists(detected_path):
        try:
            # Abrir la imagen detectada
            img_detected = Image.open(detected_path)

            # Leer las predicciones desde el archivo .txt generado por YOLO
            txt_file = os.path.join(output_directory, 'labels', os.path.basename(image_filename).replace('.jpg', '.txt'))
            
            if os.path.exists(txt_file):
                with open(txt_file, 'r') as f:
                    lines = f.readlines()

                # Filtrar solo las pelotas (ID=37) y extraer sus coordenadas
                filtered_lines = [line for line in lines if int(line.split()[0]) == 37]

                # Si hay pelotas detectadas, recortarlas de la imagen
                if filtered_lines:
                    for line in filtered_lines:
                        # Obtener las coordenadas de la caja delimitadora (bounding box)
                        data = line.split()
                        x_center, y_center, width, height = map(float, data[1:])
                        
                        # Convertir las coordenadas a píxeles (suponiendo que la imagen es del tamaño original)
                        img_width, img_height = img_detected.size
                        x_center, y_center = int(x_center * img_width), int(y_center * img_height)
                        width, height = int(width * img_width), int(height * img_height)
                        
                        # Calcular las coordenadas de la caja delimitadora
                        left = x_center - width // 2
                        top = y_center - height // 2
                        right = x_center + width // 2
                        bottom = y_center + height // 2
                        
                        # Recortar la imagen para que solo contenga la pelota detectada
                        cropped_img = img_detected.crop((left, top, right, bottom))
                        
                        # Guardar la imagen recortada
                        cropped_img.save(detected_filename)
                        print(f"El archivo detectado se guardó como: {detected_filename}")
                        break  # Solo guardar la primera pelota detectada (puedes cambiar este comportamiento si deseas más)
                else:
                    print("No se detectaron pelotas.")
            else:
                print(f"Error: No se encontró el archivo de etiquetas {txt_file}.")
        except Exception as e:
            print(f"Error al procesar la imagen detectada: {e}")
    else:
        print(f"Error: No se encontró el archivo detectado en {detected_path}.")
else:
    print("Error: No se encontró el directorio de salida.")
