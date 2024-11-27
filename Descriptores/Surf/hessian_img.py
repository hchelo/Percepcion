import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import io, color
import matplotlib.pyplot as plt

def calcular_hessiana(imagen, sigma=1):
    """
    Calcula la matriz Hessiana de una imagen utilizando filtros Gaussianos.
    
    Parámetros:
        imagen: ndarray
            Imagen de entrada en escala de grises.
        sigma: float
            Desviación estándar del filtro gaussiano.
    
    Retorna:
        hessiana: dict
            Diccionario con las componentes de la matriz Hessiana.
    """
    # Derivadas de segundo orden
    Ixx = gaussian_filter(imagen, sigma=sigma, order=[2, 0])  # Segunda derivada en x
    Iyy = gaussian_filter(imagen, sigma=sigma, order=[0, 2])  # Segunda derivada en y
    Ixy = gaussian_filter(imagen, sigma=sigma, order=[1, 1])  # Derivada cruzada
    
    # Crear un diccionario para almacenar los resultados
    hessiana = {
        "Ixx": Ixx,
        "Iyy": Iyy,
        "Ixy": Ixy
    }
    return hessiana

# Cargar la imagen pepe.jpg
imagen = io.imread("fanta.jpg")
if len(imagen.shape) == 3:  # Si la imagen tiene 3 canales, convertirla a escala de grises
    imagen = color.rgb2gray(imagen)

# Calcular la matriz Hessiana
sigma = 2
hessiana = calcular_hessiana(imagen, sigma=sigma)

# Mostrar las componentes de la matriz Hessiana
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(hessiana["Ixx"], cmap="gray")
axs[0].set_title("Ixx")
axs[1].imshow(hessiana["Iyy"], cmap="gray")
axs[1].set_title("Iyy")
axs[2].imshow(hessiana["Ixy"], cmap="gray")
axs[2].set_title("Ixy")

plt.tight_layout()
plt.show()
