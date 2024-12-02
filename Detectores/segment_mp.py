import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Cargar la imagen
input_image = cv2.imread("person2.jpg")

# Convertir imagen de BGR a RGB
rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Procesar la imagen
results = segmentation.process(rgb_image)

# Generar la máscara binaria
mask = (results.segmentation_mask > 0.5).astype(np.uint8)

# Crear una capa transparente con el color deseado (en este caso, azul)
color = (255, 0, 0)  # Azul en formato BGR
transparent_layer = np.zeros_like(input_image, dtype=np.uint8)
transparent_layer[mask == 1] = color  # Aplicar color solo en la máscara

# Mezclar la capa transparente con la imagen original
alpha = 0.5  # Nivel de transparencia (0 = completamente transparente, 1 = completamente opaco)
highlighted_image = cv2.addWeighted(input_image, 1 - alpha, transparent_layer, alpha, 0)

# Mostrar y guardar resultados
cv2.imshow("Objetos Segmentados con Máscara Transparente", highlighted_image)
cv2.imwrite("objetos_segmentados_transparente.png", highlighted_image)  # Guardar la imagen de salida
cv2.waitKey(0)
cv2.destroyAllWindows()
