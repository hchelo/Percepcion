import cv2
import numpy as np
import os
import time

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
escala = 1.0               # 0.5 = procesa aún más rápido

color_flecha = (0, 255, 255)   # amarillo (BGR)
mostrar_angulo = False         # True = escribe el ángulo sobre la original
mostrar_fps = True             # muestra los FPS en consola


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


modelos = [entrenar(archivo_1), entrenar(archivo_2), entrenar(archivo_3)]
priors = np.array([P_1, P_2, P_3])
umbrales = np.array([umbral_1, umbral_2, umbral_3])


# =====================================================================
# TABLA DE BÚSQUEDA (LUT)
# ---------------------------------------------------------------------
# Solo existen 256^3 = 16.777.216 colores BGR posibles. Se clasifican
# TODOS una sola vez al inicio y se guarda la clase de cada uno:
#   0 = fondo, 1 = verde, 2 = rojo, 3 = azul
# Luego, en cada fotograma, clasificar un píxel es solo leer la tabla.
# El resultado es idéntico al cálculo píxel a píxel.
# =====================================================================
def construir_lut():
    t0 = time.time()
    lut = np.zeros(256 ** 3, dtype=np.uint8)
    valores = np.arange(256, dtype=np.float64)
    G, R = np.meshgrid(valores, valores, indexing='ij')
    G, R = G.ravel(), R.ravel()
    filas = np.arange(65536)

    for b in range(256):                      # por bloques de 65.536 colores
        pixeles = np.column_stack([np.full(65536, b, dtype=np.float64), G, R])
        Pk = np.column_stack([verosimilitud(pixeles, *m) for m in modelos])

        clase = np.argmax(Pk * priors, axis=1)          # regla de Bayes
        supera = Pk[filas, clase] > umbrales[clase]     # umbral de su clase
        lut[b * 65536:(b + 1) * 65536] = np.where(supera, clase + 1, 0)

    print(f"LUT construida en {time.time() - t0:.1f} s")
    return lut


lut = construir_lut()

erosion_kernel = np.ones((3, 3), np.uint8)
dilation_kernel = np.ones((10, 10), np.uint8)


# =====================================================================
# FUNCIONES
# =====================================================================
def clasificar(frame):
    """Devuelve una imagen de etiquetas (0 fondo, 1, 2, 3) usando la LUT."""
    b = frame[:, :, 0].astype(np.uint32)
    g = frame[:, :, 1].astype(np.uint32)
    r = frame[:, :, 2].astype(np.uint32)
    return lut[(b << 16) | (g << 8) | r]


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


def angulo_cartesiano(direccion):
    """Ángulo cartesiano: 0° derecha (+X), 90° arriba (+Y), 180° izquierda, 270° abajo."""
    return np.degrees(np.arctan2(-direccion[1], direccion[0])) % 360


# =====================================================================
# VIDEO
# =====================================================================
if not os.path.exists(archivo_video):
    raise FileNotFoundError(f"No se encontró el video {archivo_video}")

cap = cv2.VideoCapture(archivo_video)
if not cap.isOpened():
    raise ValueError(f"No se pudo abrir el video desde {archivo_video}")

dir_suave = None
n_frames, t_inicio = 0, time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if escala != 1.0:
        frame = cv2.resize(frame, None, fx=escala, fy=escala)

    # --- Clasificación Bayes (vía LUT) ---
    etiquetas = clasificar(frame)

    mask_1 = limpiar(np.where(etiquetas == 1, 255, 0).astype(np.uint8))
    mask_2 = limpiar(np.where(etiquetas == 2, 255, 0).astype(np.uint8))
    mask_3 = limpiar(np.where(etiquetas == 3, 255, 0).astype(np.uint8))

    # --- Ventana 1: solo los colores clasificados ---
    filtrada = np.zeros_like(frame)
    filtrada[mask_1 > 0] = pintar_1
    filtrada[mask_2 > 0] = pintar_2
    filtrada[mask_3 > 0] = pintar_3

    # --- Ventana 2: original con solo la flecha ---
    con_flecha = frame.copy()

    # --- Buscar el par rojo (adelante) / azul (atrás) más cercano ---
    rojos = obtener_todos(mask_2)
    azules = obtener_todos(mask_3)

    mejor = None
    for (c_r, _) in rojos:
        for (c_a, _) in azules:
            d = np.hypot(c_r[0] - c_a[0], c_r[1] - c_a[1])
            if d <= dist_max and (mejor is None or d < mejor[0]):
                mejor = (d, c_r, c_a)

    if mejor is not None:
        _, c_rojo, c_azul = mejor
        adelante = np.array(c_rojo)
        atras = np.array(c_azul)
        centro_robot = (adelante + atras) / 2

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

            cv2.arrowedLine(con_flecha, tuple(np.int32(cola)), tuple(np.int32(punta)),
                            color_flecha, 3, tipLength=0.3)

            if mostrar_angulo:
                angulo = angulo_cartesiano(dir_suave)
                cv2.putText(con_flecha, f"{angulo:.1f} deg",
                            (int(cola[0]) + 15, int(cola[1]) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_flecha, 2)

    # --- Mostrar ---
    cv2.imshow("Colores clasificados (Bayes)", filtrada)
    cv2.imshow("Original con flecha", con_flecha)

    n_frames += 1
    if mostrar_fps and n_frames % 30 == 0:
        print(f"FPS: {n_frames / (time.time() - t_inicio):.1f}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()