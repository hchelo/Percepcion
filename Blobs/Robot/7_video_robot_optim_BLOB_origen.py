import cv2
import numpy as np
import os

# --- Configuración ---
archivo_video = 'fulbito.mp4'

if not os.path.exists(archivo_video):
    raise FileNotFoundError("No se encontró el video.")

cap = cv2.VideoCapture(archivo_video)
if not cap.isOpened():
    raise ValueError("No se pudo abrir el video.")

# Rangos BGR de los colores del robot
celeste_upper = np.array([255, 251, 180])
celeste_lower = np.array([207, 204, 96])

verde_upper = np.array([123, 255, 226])
verde_lower = np.array([44, 209, 151])

purpura_upper = np.array([223, 185, 255])
purpura_lower = np.array([151, 76, 234])

# Kernels morfológicos
kernel_er = np.ones((3,3), np.uint8)
kernel_di = np.ones((9,9), np.uint8)

# --- Función para obtener centroide ---
def centroide(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 40:
        return None, None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None, c
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), c

# --- Bucle principal ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Máscaras BGR
    mask_celeste = cv2.inRange(frame, celeste_lower, celeste_upper)
    mask_verde   = cv2.inRange(frame, verde_lower, verde_upper)
    mask_pur     = cv2.inRange(frame, purpura_lower, purpura_upper)

    # Morfología para limpiar ruido
    mask_celeste = cv2.morphologyEx(mask_celeste, cv2.MORPH_OPEN, kernel_er)
    mask_celeste = cv2.morphologyEx(mask_celeste, cv2.MORPH_CLOSE, kernel_di)

    mask_verde = cv2.morphologyEx(mask_verde, cv2.MORPH_OPEN, kernel_er)
    mask_verde = cv2.morphologyEx(mask_verde, cv2.MORPH_CLOSE, kernel_di)

    mask_pur = cv2.morphologyEx(mask_pur, cv2.MORPH_OPEN, kernel_er)
    mask_pur = cv2.morphologyEx(mask_pur, cv2.MORPH_CLOSE, kernel_di)

    # Centroides
    centro_cel, cont_cel = centroide(mask_celeste)
    centro_pur, cont_pur = centroide(mask_pur)

    # --- Imagen filtrada ---
    filtrada = np.zeros_like(frame)
    filtrada[np.where(mask_celeste > 0)] = (255, 0, 0)
    filtrada[np.where(mask_verde > 0)]   = (0, 255, 0)
    filtrada[np.where(mask_pur > 0)]     = (255, 0, 255)

    # Dibujar flecha en filtrada
    if centro_cel is not None and centro_pur is not None:
        start = centro_cel
        end = centro_pur
        vx = end[0] - start[0]
        vy = end[1] - start[1]
        escala = 2.5
        end_arrow = (start[0] + int(vx*escala), start[1] + int(vy*escala))
        cv2.arrowedLine(filtrada, start, end_arrow, (0,0,255), 3, tipLength=0.25)

    # --- Frame sin flecha ---
    sin_flecha = frame.copy()

    # --- Frame con flecha ---
    con_flecha = frame.copy()
    if centro_cel is not None and centro_pur is not None:
        cv2.arrowedLine(con_flecha, start, end_arrow, (0,0,255), 3, tipLength=0.25)

    # --- Mostrar ventanas ---
    cv2.imshow("Original", frame)
    cv2.imshow("Filtrada", filtrada)
    #cv2.imshow("Video SIN Flecha (BGR)", sin_flecha)
    cv2.imshow("Video CON Flecha (BGR)", con_flecha)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
