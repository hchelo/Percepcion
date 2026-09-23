import cv2
import numpy as np
import glob
import os

# =====================================================================
# CONFIGURACIÓN
# =====================================================================
carpeta_entrenamiento = 'entrenamiento'   # carpeta con tus 8 imágenes
extensiones = ('*.jpg', '*.jpeg', '*.png', '*.JPG')

# Mismos nombres de archivo que usa bayes_3_colores.py
clases = [
    {'nombre': 'Color 1', 'archivo': 'valores_rgb_color1.txt'},
    {'nombre': 'Color 2', 'archivo': 'valores_rgb_color2.txt'},
    {'nombre': 'Color 3', 'archivo': 'valores_rgb_color3.txt'},
    # Descomenta si quieres el fondo como cuarta clase:
    # {'nombre': 'Fondo',   'archivo': 'valores_rgb_fondo.txt'},
]

guardar_unicos = False   # True = elimina píxeles repetidos (como tu archivo "unicos")
max_ancho = 1200         # reduce la ventana si la imagen es muy grande

# =====================================================================
# EXTRACCIÓN
# =====================================================================
rutas = []
for ext in extensiones:
    rutas += glob.glob(os.path.join(carpeta_entrenamiento, ext))
rutas = sorted(set(rutas))
print(f"{len(rutas)} imagenes encontradas")

muestras = {c['archivo']: [] for c in clases}

for ruta in rutas:
    imagen = cv2.imread(ruta)
    if imagen is None:
        print(f"No se pudo leer {ruta}, se omite")
        continue

    # Escala solo para mostrar; los píxeles se toman de la imagen original
    escala = min(1.0, max_ancho / imagen.shape[1])
    vista = cv2.resize(imagen, None, fx=escala, fy=escala)

    for c in clases:
        titulo = f"{os.path.basename(ruta)} - {c['nombre']} (ENTER por recuadro, ESC para terminar)"
        rois = cv2.selectROIs(titulo, vista, showCrosshair=False)
        cv2.destroyWindow(titulo)

        for (x, y, w, h) in rois:
            if w == 0 or h == 0:
                continue
            x0, y0 = int(x / escala), int(y / escala)
            x1, y1 = int((x + w) / escala), int((y + h) / escala)
            recorte = imagen[y0:y1, x0:x1].reshape(-1, 3)
            muestras[c['archivo']].append(recorte)

        total = sum(len(m) for m in muestras[c['archivo']])
        print(f"  {os.path.basename(ruta)} | {c['nombre']}: {len(rois)} recuadros, acumulado {total} px")

# =====================================================================
# GUARDADO (mismo formato "b, g, r" que lee el clasificador)
# =====================================================================
for c in clases:
    if not muestras[c['archivo']]:
        print(f"ATENCIÓN: {c['nombre']} no tiene muestras")
        continue
    datos = np.vstack(muestras[c['archivo']])
    if guardar_unicos:
        datos = np.unique(datos, axis=0)
    np.savetxt(c['archivo'], datos, fmt='%d', delimiter=', ')
    print(f"{c['archivo']}: {len(datos)} píxeles guardados")