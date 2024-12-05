import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Cargar la imagen
input_image = cv2.imread("gestos/person_closed.png")

# Convertir imagen de BGR a RGB
rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Procesar la imagen
results = segmentation.process(rgb_image)

# Generar la máscara binaria
mask = (results.segmentation_mask > 0.5).astype(np.uint8)

# Crear una imagen con fondo verde
green_background = np.zeros_like(input_image, dtype=np.uint8)
green_background[:] = (0, 255, 0)  # Verde en formato BGR

# Combinar el primer plano con el fondo verde
output_image = np.where(mask[..., None] == 1, input_image, green_background)

# Mostrar y guardar resultados
cv2.imshow("Imagen con Fondo Verde", output_image)
cv2.imwrite("imagen_fondo_verde.png", output_image)  # Guardar la imagen de salida
cv2.waitKey(0)
cv2.destroyAllWindows()
