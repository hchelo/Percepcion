import cv2
import numpy as np

# Ruta al archivo de video
archivo_video = 'fulbito.mp4'

cap = cv2.VideoCapture(archivo_video)
if not cap.isOpened():
    raise ValueError(f"No se pudo abrir el video desde {archivo_video}")

# --- Rangos de colores (ajusta según tu video) ---
celeste_upper = np.array([255, 251, 180])
celeste_lower = np.array([207, 204, 96])

verde_upper = np.array([123, 255, 226])
verde_lower = np.array([44, 209, 151])

# Aquí está tu púrpura (antes "rojo")
purpura_upper = np.array([223, 185, 255])
purpura_lower = np.array([151, 76, 234])

erosion_kernel = np.ones((3, 3), np.uint8)
dilation_kernel = np.ones((10, 10), np.uint8)

# Función para obtener centro de una máscara
def obtener_centro(mask):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contornos) == 0:
        return None

    cont = max(contornos, key=cv2.contourArea)
    M = cv2.moments(cont)

    if M["m00"] == 0:
        return None

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return (cX, cY), cont


while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    # --- Crear máscaras ---
    celeste_mask = cv2.inRange(img, celeste_lower, celeste_upper)
    verde_mask   = cv2.inRange(img, verde_lower, verde_upper)
    purpura_mask = cv2.inRange(img, purpura_lower, purpura_upper)

    # --- Recolorar según tus reglas ---
    img[np.where(celeste_mask == 255)] = (255, 0, 0)
    img[np.where(verde_mask == 255)]   = (0, 255, 0)
    img[np.where(purpura_mask == 255)] = (255, 0, 255)

    # Resto a negro
    negro_mask = ~(celeste_mask | verde_mask | purpura_mask)
    img[np.where(negro_mask)] = (0, 0, 0)

    # --- Erosión y dilatación ---
    img = cv2.erode(img, erosion_kernel, iterations=1)
    img = cv2.dilate(img, dilation_kernel, iterations=1)

    # --- Obtener centros ---
    celeste_data = obtener_centro(celeste_mask)
    purpura_data = obtener_centro(purpura_mask)

    img_draw = img.copy()

    # Dibujar los puntos detectados
    if celeste_data:
        (cx_c, cy_c), cont_c = celeste_data
        cv2.circle(img_draw, (cx_c, cy_c), 6, (255, 255, 0), -1)
        cv2.drawContours(img_draw, [cont_c], -1, (255, 255, 0), 2)

    if purpura_data:
        (cx_p, cy_p), cont_p = purpura_data
        cv2.circle(img_draw, (cx_p, cy_p), 6, (255, 0, 255), -1)
        cv2.drawContours(img_draw, [cont_p], -1, (255, 0, 255), 2)

    # --- Dibujar flecha de orientación ---
    if celeste_data and purpura_data:
        start = (cx_c, cy_c)   # centro del robot
        end = (cx_p, cy_p)     # cabeza del robot

        # Crear flecha más larga
        vx = cx_p - cx_c
        vy = cy_p - cy_c
        escala = 3  # <- aumenta el largo si lo deseas
        end_arrow = (cx_c + int(vx * escala), cy_c + int(vy * escala))

        cv2.arrowedLine(img_draw, start, end_arrow, (255, 255, 255), 4, tipLength=0.3)

    # Mostrar imagen procesada
    cv2.imshow('Video Procesado + Direccion', img_draw)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
