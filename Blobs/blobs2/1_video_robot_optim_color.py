import cv2
import numpy as np

# Ruta al archivo de video
archivo_video = 'fulbito.mp4'  # Actualiza esta línea con la ruta correcta del archivo de video

# Abrir el archivo de video
cap = cv2.VideoCapture(archivo_video)
if not cap.isOpened():
    raise ValueError(f"No se pudo abrir el video desde {archivo_video}")

# Definir el kernel para la erosión y dilatación
kernel_erosion = np.ones((3, 3), np.uint8)  # Kernel de 3x3 para la erosión
celeste_upper = np.array([255, 251, 180])
celeste_lower = np.array([207, 204, 96])  # [B, G, R]

verde_upper = np.array([123, 255, 226])
verde_lower = np.array([44, 209, 151])  # [B, G, R]

rojo_upper = np.array([223, 185, 255])
rojo_lower = np.array([151, 76, 234])  # [B, G, R]
# Leer el primer fotograma del video
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break  # Si no hay más fotogramas, salir del ciclo

    # Convertir la imagen a escala de grises
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Crear máscaras para cada rango
    celeste_mask = cv2.inRange(img, celeste_lower, celeste_upper)
    verde_mask = cv2.inRange(img, verde_lower, verde_upper)
    rojo_mask = cv2.inRange(img, rojo_lower, rojo_upper)
    
    # Aplicar los colores basados en las máscaras
    img[np.where(celeste_mask == 255)] = (255, 0, 0)  # Celeste
    img[np.where(verde_mask == 255)] = (0, 255, 0)    # Verde
    img[np.where(rojo_mask == 255)] = (0, 0, 255)    # Otro

        # El resto se pone negro
    negro_mask = ~(celeste_mask | verde_mask | rojo_mask)
    img[np.where(negro_mask)] = (0, 0, 0)
            
    erosion_kernel = np.ones((3, 3), np.uint8)
    dilation_kernel = np.ones((7, 7), np.uint8)

    # Realizar erosión y dilatación en una sola operación si aplica
    img = cv2.erode(img, erosion_kernel, iterations=1)
    img = cv2.dilate(img, dilation_kernel, iterations=1)

    cv2.imshow('Video Procesado', img)  # Convertir de RGB a BGR para mostrar
    # Esperar por una tecla para avanzar al siguiente fotograma (presionar 'q' para salir)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el objeto de captura y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
