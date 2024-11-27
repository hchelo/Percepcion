import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float
from skimage.filters import frangi, threshold_otsu
from scipy.ndimage import binary_erosion, binary_dilation

# Cargar la imagen
imagen_path = "vessel.png"  # Cambia esto por la ruta de tu imagen
imagen = io.imread(imagen_path)

# Convertir a escala de grises si es necesario
if len(imagen.shape) == 3:  # Si la imagen es RGB
    imagen_gray = color.rgb2gray(imagen)
else:
    imagen_gray = imagen

# Convertir la imagen a rango flotante
imagen_gray = img_as_float(imagen_gray)

# Aplicar el filtro de Frangi
imagen_frangi = frangi(imagen_gray, scale_range=(1, 10), scale_step=2)

# Calcular el umbral adaptativo usando Otsu
umbral_otsu = threshold_otsu(imagen_frangi)
imagen_umbralizada = imagen_frangi > umbral_otsu

# Definir kernels para erosión y dilatación
kernel_erosion = np.ones((2, 2), dtype=bool)  # Kernel 3x3 para erosión
kernel_dilation = np.ones((3, 3), dtype=bool)  # Kernel 5x5 para dilatación

# Aplicar dilatación y luego erosión
imagen_dilatada = binary_erosion(imagen_umbralizada, structure=kernel_erosion)
imagen_refinada = binary_dilation(imagen_dilatada, structure=kernel_dilation)

# Mostrar los resultados
fig, ax = plt.subplots(1, 4, figsize=(20, 5))

ax[0].imshow(imagen_gray, cmap="gray")
ax[0].set_title("Imagen Original")
ax[0].axis("off")

ax[1].imshow(imagen_frangi, cmap="gray")
ax[1].set_title("Filtro Frangi")
ax[1].axis("off")

ax[2].imshow(imagen_umbralizada, cmap="gray")
ax[2].set_title("Umbral Adaptativo (Otsu)")
ax[2].axis("off")

ax[3].imshow(imagen_refinada, cmap="gray")
ax[3].set_title("Imagen Refinada")
ax[3].axis("off")

plt.tight_layout()
plt.show()
