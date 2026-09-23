import cv2
import numpy as np
import glob
import os

# =====================================================================
# CONFIGURACIÓN
# =====================================================================
carpeta_prueba = 'prueba'          # carpeta con las 8 imágenes de prueba
carpeta_resultados = 'resultados'  # aquí se guardan las salidas
os.makedirs(carpeta_resultados, exist_ok=True)

extensiones = ('*.jpg', '*.jpeg', '*.png', '*.JPG')

# Archivos generados por extraer_muestras.py (con las fotos de entrenamiento)
archivo_1 = 'valores_rgb_color1.txt'
archivo_2 = 'valores_rgb_color2.txt'
archivo_3 = 'valores_rgb_color3.txt'

# Color con el que se pinta cada clase (BGR)
pintar_1 = [0, 255, 0]     # Verde
pintar_2 = [0, 0, 255]     # Rojo
pintar_3 = [255, 0, 0]     # Azul

# Probabilidades a priori (iguales para las 3 clases)
P_1 = 1 / 3
P_2 = 1 / 3
P_3 = 1 / 3

# Umbral para decidir fondo
umbral = 0.0000005

mostrar_ventanas = True    # False = solo guarda los resultados, sin abrir ventanas


# =====================================================================
# FUNCIONES
# =====================================================================
def entrenar(nombre_archivo):
    valores_rgb = []
    with open(nombre_archivo, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                rgb = tuple(map(int, line.split(', ')))
                valores_rgb.append(rgb)

    MatrizRGB = np.array(valores_rgb, dtype=np.float64)

    media = np.mean(MatrizRGB, axis=0)
    covarianza = np.cov(MatrizRGB, rowvar=False)
    cova = np.diag(np.diag(covarianza))          # Covarianza diagonalizada
    cova = cova + 1e-6 * np.eye(3)               # Evita determinante cero

    deter = np.linalg.det(cova)
    inversa = np.linalg.inv(cova)
    constante = 1 / np.sqrt(((2 * np.pi) ** 3) * deter)

    print(f"{nombre_archivo}: {len(MatrizRGB)} muestras, media = {np.round(media, 1)}")
    return media, inversa, constante


def verosimilitud(pixeles, media, inversa, constante):
    dif_pixel_media = pixeles - media
    Sk = np.sum(dif_pixel_media @ inversa * dif_pixel_media, axis=1)
    return constante * np.exp(-Sk / 2)


kernel_diamante = np.array([[0, 0, 1, 0, 0],
                            [0, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 0],
                            [0, 0, 1, 0, 0]], dtype=np.uint8)
kernel_cierre = np.ones((7, 7), np.uint8)


def limpiar(mascara):
    """Erosión + dilatación + cierre."""
    m = cv2.erode(mascara, kernel_diamante, iterations=1)
    m = cv2.dilate(m, kernel_diamante, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_cierre)
    return m


# =====================================================================
# ENTRENAMIENTO DE LAS 3 CLASES
# =====================================================================
media_1, inversa_1, constante_1 = entrenar(archivo_1)
media_2, inversa_2, constante_2 = entrenar(archivo_2)
media_3, inversa_3, constante_3 = entrenar(archivo_3)

# =====================================================================
# IMÁGENES DE PRUEBA
# =====================================================================
imagenes_path = []
for ext in extensiones:
    imagenes_path += glob.glob(os.path.join(carpeta_prueba, ext))
imagenes_path = sorted(set(imagenes_path))

print(f"\n{len(imagenes_path)} imágenes de prueba en '{carpeta_prueba}'")

# =====================================================================
# CLASIFICACIÓN DE CADA IMAGEN
# =====================================================================
for imagen_path in imagenes_path:
    imagen = cv2.imread(imagen_path)
    if imagen is None:
        print(f"No se pudo leer {imagen_path}")
        continue

    alto, ancho = imagen.shape[:2]
    imagen_rgb_float = np.float64(imagen).reshape((-1, 3))

    # Verosimilitud p(x | color k)
    Pk_1 = verosimilitud(imagen_rgb_float, media_1, inversa_1, constante_1)
    Pk_2 = verosimilitud(imagen_rgb_float, media_2, inversa_2, constante_2)
    Pk_3 = verosimilitud(imagen_rgb_float, media_3, inversa_3, constante_3)

    # Regla de Bayes: gana la clase con mayor p(x|k) * P(k)
    posteriores = np.column_stack([Pk_1 * P_1, Pk_2 * P_2, Pk_3 * P_3])
    clase = np.argmax(posteriores, axis=1)

    # Fondo: ninguna clase supera el umbral
    Pk_max = np.max(np.column_stack([Pk_1, Pk_2, Pk_3]), axis=1)
    fondo = Pk_max <= umbral

    # Imagen filtrada (clasificación cruda)
    new_RGB_flat = np.zeros_like(imagen_rgb_float, dtype=np.uint8)
    new_RGB_flat[(clase == 0) & ~fondo] = pintar_1
    new_RGB_flat[(clase == 1) & ~fondo] = pintar_2
    new_RGB_flat[(clase == 2) & ~fondo] = pintar_3
    new_RGB = new_RGB_flat.reshape(imagen.shape)

    # Máscaras binarias de cada color
    mask_1 = (((clase == 0) & ~fondo).reshape(alto, ancho) * 255).astype(np.uint8)
    mask_2 = (((clase == 1) & ~fondo).reshape(alto, ancho) * 255).astype(np.uint8)
    mask_3 = (((clase == 2) & ~fondo).reshape(alto, ancho) * 255).astype(np.uint8)

    # Morfología
    final_1 = limpiar(mask_1)
    final_2 = limpiar(mask_2)
    final_3 = limpiar(mask_3)

    # Pintar el resultado sobre la imagen original
    imagen_resultado = imagen.copy()
    imagen_resultado[final_1 > 0] = pintar_1
    imagen_resultado[final_2 > 0] = pintar_2
    imagen_resultado[final_3 > 0] = pintar_3

    # Porcentaje de píxeles de cada clase (útil para comparar imágenes)
    total = alto * ancho
    print(f"{os.path.basename(imagen_path)}: "
          f"C1 {100 * np.count_nonzero(final_1) / total:.1f}% | "
          f"C2 {100 * np.count_nonzero(final_2) / total:.1f}% | "
          f"C3 {100 * np.count_nonzero(final_3) / total:.1f}%")

    # Guardar resultados
    base = os.path.splitext(os.path.basename(imagen_path))[0]
    cv2.imwrite(os.path.join(carpeta_resultados, f'{base}_filtrada.png'), new_RGB)
    cv2.imwrite(os.path.join(carpeta_resultados, f'{base}_resultado.png'), imagen_resultado)
    cv2.imwrite(os.path.join(carpeta_resultados, f'{base}_color1.png'), final_1)
    cv2.imwrite(os.path.join(carpeta_resultados, f'{base}_color2.png'), final_2)
    cv2.imwrite(os.path.join(carpeta_resultados, f'{base}_color3.png'), final_3)

    # Mostrar
    if mostrar_ventanas:
        cv2.imshow('Original', imagen)
        cv2.imshow('Filtrada Bayes', new_RGB)
        #cv2.imshow('Color 1', final_1)
        #cv2.imshow('Color 2', final_2)
        #cv2.imshow('Color 3', final_3)
        cv2.imshow('Resultado', imagen_resultado)
        print("   presiona una tecla para la siguiente (ESC para salir)")
        tecla = cv2.waitKey(0)
        if tecla == 27:
            break

cv2.destroyAllWindows()
print(f"\nResultados guardados en '{carpeta_resultados}'")