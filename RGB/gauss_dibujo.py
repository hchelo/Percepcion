import math
import numpy as np

# Crear un vector con los números del 1 al 100
vector = np.array([i for i in range(1, 101)])

# Calcular la media (mu) y la desviación estándar (sigma)
mu = np.mean(vector)
sigma = np.std(vector, ddof=0)  # ddof=0 para la desviación estándar poblacional

# Definir la función gaussiana
def gaussiana(x, mu, sigma):
    return (1 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# Aplicar la función gaussiana a cada número del vector
gaussian_vector = [gaussiana(x, mu, sigma) for x in vector]

# Imprimir los valores calculados
print(f"Media (mu): {mu}")
print(f"Desviación estándar (sigma): {sigma}")
print("Valores gaussianos:", gaussian_vector)

# Imprimir los valores calculados
print(f"Media (mu): {mu}")
print(f"Desviación estándar (sigma): {sigma}")
print("Valores gaussianos:", gaussian_vector)

import matplotlib.pyplot as plt

plt.plot(vector, gaussian_vector)
plt.xlabel("x")
plt.ylabel("Gauss(x)")
plt.title("Distribución Gaussiana")
plt.grid()
plt.show()