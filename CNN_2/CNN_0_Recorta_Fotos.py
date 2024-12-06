import os
import cv2
from mtcnn import MTCNN

# Directorios de la base de datos
dataset_dir = 'Feret_train/'  # Cambia esto con la ruta de tu base de datos
hombres_dir = os.path.join(dataset_dir, 'hombres')
mujeres_dir = os.path.join(dataset_dir, 'mujeres')

# Crear las subcarpetas para almacenar los recortes de rostros
hombres_reco_dir = os.path.join(dataset_dir, 'hombres_reco')
mujeres_reco_dir = os.path.join(dataset_dir, 'mujeres_reco')

os.makedirs(hombres_reco_dir, exist_ok=True)
os.makedirs(mujeres_reco_dir, exist_ok=True)

# Inicializar el detector de MTCNN
detector = MTCNN()

# Función para detectar y guardar rostros
def detect_and_save_faces(image_path, save_dir, label):
    # Cargar imagen
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detectar caras
    results = detector.detect_faces(rgb_image)

    # Guardar los recortes de los rostros detectados
    for i, result in enumerate(results):
        x, y, width, height = result['box']
        face = image[y:y+height, x:x+width]

        # Guardar el recorte en el directorio correspondiente
        if label == 'hombre':
            save_path = os.path.join(save_dir, f"hombre_{os.path.basename(image_path).split('.')[0]}_{i}.jpg")
        elif label == 'mujer':
            save_path = os.path.join(save_dir, f"mujer_{os.path.basename(image_path).split('.')[0]}_{i}.jpg")
        
        cv2.imwrite(save_path, face)

# Procesar las imágenes de hombres
for filename in os.listdir(hombres_dir):
    if filename.lower().endswith(('jpg', 'jpeg', 'png')):
        img_path = os.path.join(hombres_dir, filename)
        detect_and_save_faces(img_path, hombres_reco_dir, 'hombre')

# Procesar las imágenes de mujeres
for filename in os.listdir(mujeres_dir):
    if filename.lower().endswith(('jpg', 'jpeg', 'png')):
        img_path = os.path.join(mujeres_dir, filename)
        detect_and_save_faces(img_path, mujeres_reco_dir, 'mujer')

print("Recorte de rostros completado.")
