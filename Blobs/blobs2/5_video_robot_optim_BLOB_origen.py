import cv2
import numpy as np

# Ruta al archivo de video
archivo_video = 'fulbito.mp4'  # Actualiza esta línea con la ruta correcta del archivo de video

# Abrir el archivo de video
cap = cv2.VideoCapture(archivo_video)
if not cap.isOpened():
    raise ValueError(f"No se pudo abrir el video desde {archivo_video}")

# Definir los rangos de colores en [B, G, R]
celeste_upper = np.array([255, 251, 180])
celeste_lower = np.array([207, 204, 96])

verde_upper = np.array([123, 255, 226])
verde_lower = np.array([44, 209, 151])

rojo_upper = np.array([223, 185, 255])
rojo_lower = np.array([151, 76, 234])

# Bucle para procesar cada fotograma
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break  # Salir del bucle si no hay más fotogramas

    # Crear máscaras para los colores
    celeste_mask = cv2.inRange(img, celeste_lower, celeste_upper)
    verde_mask = cv2.inRange(img, verde_lower, verde_upper)
    rojo_mask = cv2.inRange(img, rojo_lower, rojo_upper)

    # Crear la imagen binaria para los colores detectados
    combined_mask = celeste_mask | verde_mask | rojo_mask
    result = np.zeros_like(img)
    result[combined_mask == 255] = (255, 255, 255)  # Blanco
    result[combined_mask == 0] = (0, 0, 0)          # Negro

    # Erosión y dilatación
    erosion_kernel = np.ones((5, 5), np.uint8)
    dilation_kernel = np.ones((35, 35), np.uint8)
    result = cv2.erode(result, erosion_kernel, iterations=1)
    result = cv2.dilate(result, dilation_kernel, iterations=1)

    # Convertir a escala de grises y aplicar binarización
    im_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(im_gray, 57, 255, cv2.THRESH_BINARY)

    # Detectar contornos
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear una copia de la imagen original para superponer los contornos
    img_with_contours = img.copy()

    # Dibujar contornos y numerar blobs
    blob_counter = 1
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Dibujar los contornos en rojo
            cv2.drawContours(img_with_contours, [contour], -1, (0, 0, 255), 2)
            cv2.line(img_with_contours, (cX - 10, cY), (cX + 10, cY), (0, 0, 255), 2)
            cv2.line(img_with_contours, (cX, cY - 10), (cX, cY + 10), (0, 0, 255), 2)
            cv2.putText(img_with_contours, str(blob_counter), (cX - 15, cY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            blob_counter += 1

    # Mostrar el fotograma original, procesado, y con contornos superpuestos
    cv2.imshow("Original", img)
    cv2.imshow("Procesado", result)
    cv2.imshow("Contornos sobre Original", img_with_contours)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
