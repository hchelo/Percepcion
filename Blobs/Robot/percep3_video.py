import cv2
import numpy as np

# =====================================================================
# CONFIGURACIÓN
# =====================================================================
archivo_video = 'fulbito.mp4'

# Archivos entrenados (color 1 = verde, 2 = rojo, 3 = azul)
archivo_1 = 'valores_rgb_color1.txt'
archivo_2 = 'valores_rgb_color2.txt'
archivo_3 = 'valores_rgb_color3.txt'

# Color con el que se pinta cada clase (BGR)
pintar_1 = [0, 255, 0]     # Verde
pintar_2 = [0, 0, 255]     # Rojo
pintar_3 = [255, 0, 0]     # Azul

# Probabilidades a priori
P_1 = 1 / 3
P_2 = 1 / 3
P_3 = 1 / 3

# Umbral por clase (súbelo si esa clase agarra fondo)
umbral_1 = 0.000005
umbral_2 = 0.0000005
umbral_3 = 0.0000005

escala = 1.0   # 0.5 = procesa a la mitad de tamaño (más rápido)


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

# Kernels (los mismos de tu código de video)
erosion_kernel = np.ones((3, 3), np.uint8)
dilation_kernel = np.ones((10, 10), np.uint8)


def limpiar(mascara):
    m = cv2.erode(mascara, erosion_kernel, iterations=1)
    m = cv2.dilate(m, dilation_kernel, iterations=1)
    return m


# =====================================================================
# VIDEO
# =====================================================================
cap = cv2.VideoCapture(archivo_video)
if not cap.isOpened():
    raise ValueError(f"No se pudo abrir el video desde {archivo_video}")

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    if escala != 1.0:
        img = cv2.resize(img, None, fx=escala, fy=escala)

    alto, ancho = img.shape[:2]
    pixeles = np.float64(img).reshape((-1, 3))

    # Verosimilitudes
    Pk_1 = verosimilitud(pixeles, media_1, inversa_1, constante_1)
    Pk_2 = verosimilitud(pixeles, media_2, inversa_2, constante_2)
    Pk_3 = verosimilitud(pixeles, media_3, inversa_3, constante_3)

    # Bayes: clase con mayor p(x|k) * P(k)
    clase = np.argmax(np.column_stack([Pk_1 * P_1, Pk_2 * P_2, Pk_3 * P_3]), axis=1)

    # Cada clase debe superar su propio umbral; si no, es fondo
    m1 = (clase == 0) & (Pk_1 > umbral_1)
    m2 = (clase == 1) & (Pk_2 > umbral_2)
    m3 = (clase == 2) & (Pk_3 > umbral_3)

    mask_1 = limpiar((m1.reshape(alto, ancho) * 255).astype(np.uint8))
    mask_2 = limpiar((m2.reshape(alto, ancho) * 255).astype(np.uint8))
    mask_3 = limpiar((m3.reshape(alto, ancho) * 255).astype(np.uint8))

    # Imagen filtrada: fondo negro y cada color pintado
    filtrada = np.zeros_like(img)
    filtrada[mask_1 > 0] = pintar_1
    filtrada[mask_2 > 0] = pintar_2
    filtrada[mask_3 > 0] = pintar_3

    cv2.imshow('Video Procesado', filtrada)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()