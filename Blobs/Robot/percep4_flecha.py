import cv2
import numpy as np

# =====================================================================
# CONFIGURACIÓN
# =====================================================================
archivo_video = 'fulbito.mp4'

# Archivos entrenados
archivo_1 = 'valores_rgb_color1.txt'   # verde
archivo_2 = 'valores_rgb_color2.txt'   # rojo  -> ADELANTE del robot
archivo_3 = 'valores_rgb_color3.txt'   # azul  -> ATRÁS del robot

pintar_1 = (0, 255, 0)     # Verde (BGR)
pintar_2 = (0, 0, 255)     # Rojo
pintar_3 = (255, 0, 0)     # Azul

P_1 = P_2 = P_3 = 1 / 3    # Probabilidades a priori

umbral_1 = 0.000001        # súbelo si esa clase agarra fondo
umbral_2 = 0.0000005
umbral_3 = 0.0000005

area_minima = 50           # contornos más chicos se ignoran (ruido)
dist_max = 150             # distancia máx. (px) entre rojo y azul del mismo robot
alfa = 0.3                 # suavizado de la dirección (bajo = más estable)
escala_flecha = 1.5        # largo de la flecha
escala = 1.0               # 0.5 = procesa más rápido


# =====================================================================
# ENTRENAMIENTO
# =====================================================================
def entrenar(nombre_archivo):
    valores_rgb = []
    with open(nombre_archivo, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                valores_rgb.append(tuple(map(int, line.split(', '))))

    MatrizRGB = np.array(valores_rgb, dtype=np.float64)
    media = np.mean(MatrizRGB, axis=0)
    covarianza = np.cov(MatrizRGB, rowvar=False)
    cova = np.diag(np.diag(covarianza)) + 1e-6 * np.eye(3)

    deter = np.linalg.det(cova)
    inversa = np.linalg.inv(cova)
    constante = 1 / np.sqrt(((2 * np.pi) ** 3) * deter)

    print(f"{nombre_archivo}: {len(MatrizRGB)} muestras, media = {np.round(media, 1)}")
    return media, inversa, constante


def verosimilitud(pixeles, media, inversa, constante):
    dif = pixeles - media
    Sk = np.sum(dif @ inversa * dif, axis=1)
    return constante * np.exp(-Sk / 2)


media_1, inversa_1, constante_1 = entrenar(archivo_1)
media_2, inversa_2, constante_2 = entrenar(archivo_2)
media_3, inversa_3, constante_3 = entrenar(archivo_3)

erosion_kernel = np.ones((3, 3), np.uint8)
dilation_kernel = np.ones((10, 10), np.uint8)


# =====================================================================
# FUNCIONES
# =====================================================================
def limpiar(mascara):
    m = cv2.erode(mascara, erosion_kernel, iterations=1)
    m = cv2.dilate(m, dilation_kernel, iterations=1)
    return m


def obtener_todos(mask):
    """Lista de (centro, contorno) de todos los blobs válidos."""
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    salida = []
    for c in contornos:
        if cv2.contourArea(c) < area_minima:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        salida.append(((M["m10"] / M["m00"], M["m01"] / M["m00"]), c))
    return salida


def angulo_reloj(direccion):
    """0° derecha, 90° abajo, 180° izquierda, 270° arriba (sentido horario)."""
    return np.degrees(np.arctan2(-direccion[1], direccion[0])) % 360


# =====================================================================
# VIDEO
# =====================================================================
cap = cv2.VideoCapture(archivo_video)
if not cap.isOpened():
    raise ValueError(f"No se pudo abrir el video desde {archivo_video}")

dir_suave = None

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    if escala != 1.0:
        img = cv2.resize(img, None, fx=escala, fy=escala)

    alto, ancho = img.shape[:2]
    pixeles = np.float64(img).reshape((-1, 3))

    # --- Clasificación Bayes ---
    Pk_1 = verosimilitud(pixeles, media_1, inversa_1, constante_1)
    Pk_2 = verosimilitud(pixeles, media_2, inversa_2, constante_2)
    Pk_3 = verosimilitud(pixeles, media_3, inversa_3, constante_3)

    clase = np.argmax(np.column_stack([Pk_1 * P_1, Pk_2 * P_2, Pk_3 * P_3]), axis=1)

    m1 = (clase == 0) & (Pk_1 > umbral_1)
    m2 = (clase == 1) & (Pk_2 > umbral_2)
    m3 = (clase == 2) & (Pk_3 > umbral_3)

    mask_1 = limpiar((m1.reshape(alto, ancho) * 255).astype(np.uint8))
    mask_2 = limpiar((m2.reshape(alto, ancho) * 255).astype(np.uint8))
    mask_3 = limpiar((m3.reshape(alto, ancho) * 255).astype(np.uint8))

    # --- Imagen filtrada ---
    img_draw = np.zeros_like(img)
    img_draw[mask_1 > 0] = pintar_1
    img_draw[mask_2 > 0] = pintar_2
    img_draw[mask_3 > 0] = pintar_3

    # --- Buscar el par rojo (adelante) / azul (atrás) más cercano ---
    rojos = obtener_todos(mask_2)
    azules = obtener_todos(mask_3)

    mejor = None
    for (c_r, cont_r) in rojos:
        for (c_a, cont_a) in azules:
            d = np.hypot(c_r[0] - c_a[0], c_r[1] - c_a[1])
            if d <= dist_max and (mejor is None or d < mejor[0]):
                mejor = (d, c_r, cont_r, c_a, cont_a)

    if mejor is not None:
        _, c_rojo, cont_rojo, c_azul, cont_azul = mejor
        adelante = np.array(c_rojo)
        atras = np.array(c_azul)
        centro_robot = (adelante + atras) / 2

        # --- Bloque: rojo + azul (+ verde si está cerca) ---
        partes = [cont_rojo, cont_azul]
        for (c_v, cont_v) in obtener_todos(mask_1):
            if np.hypot(c_v[0] - centro_robot[0], c_v[1] - centro_robot[1]) <= dist_max:
                partes.append(cont_v)

        rect = cv2.minAreaRect(np.vstack(partes))
        caja = np.int32(cv2.boxPoints(rect))
        cv2.drawContours(img_draw, [caja], -1, (0, 255, 255), 2)

        # --- Dirección atrás -> adelante, normalizada y suavizada ---
        direccion = adelante - atras
        norma = np.linalg.norm(direccion)
        if norma > 0:
            direccion = direccion / norma
            if dir_suave is None:
                dir_suave = direccion
            else:
                dir_suave = alfa * direccion + (1 - alfa) * dir_suave
                dir_suave = dir_suave / np.linalg.norm(dir_suave)

        if dir_suave is not None:
            largo = max(norma, 30) * escala_flecha
            cola = centro_robot
            punta = centro_robot + dir_suave * largo

            cv2.circle(img_draw, tuple(np.int32(adelante)), 6, (255, 255, 255), -1)
            cv2.circle(img_draw, tuple(np.int32(atras)), 6, (120, 120, 120), -1)
            cv2.arrowedLine(img_draw, tuple(np.int32(cola)), tuple(np.int32(punta)),
                            (255, 255, 255), 4, tipLength=0.3)

            angulo = angulo_reloj(dir_suave)
            cv2.putText(img_draw, f"{angulo:.1f} deg",
                        (int(cola[0]) + 15, int(cola[1]) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Video Procesado + Direccion', img_draw)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()