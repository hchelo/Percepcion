import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
imagen = cv2.imread("focos.jpg")

# Redimensionar la imagen (por ejemplo, a la mitad de su tamaño)
imagen_reducida = cv2.resize(imagen, (0, 0), fx=0.5, fy=0.5)

# Convertir la imagen reducida a escala de grises
imagen_gris = cv2.cvtColor(imagen_reducida, cv2.COLOR_BGR2GRAY)

# Aplicar el filtro Canny para detectar bordes
bordes = cv2.Canny(imagen_gris, 100, 200)

# Aplicar la Transformada de Hough para detectar líneas
lineas = cv2.HoughLines(bordes, rho=1, theta=np.pi/180, threshold=50)

# Crear una copia de la imagen para dibujar las líneas
dibujada = imagen_reducida.copy()

# Dibujar las líneas detectadas en la imagen
if lineas is not None:
    for linea in lineas:
        rho, theta = linea[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(dibujada, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Graficar el espacio de Hough
hough_space = np.zeros((180, int(bordes.shape[0] + bordes.shape[1])), dtype=np.uint8)
if lineas is not None:
    for linea in lineas:
        rho, theta = linea[0]
        theta_deg = int(np.degrees(theta))
        rho_index = int(rho + (bordes.shape[0] + bordes.shape[1]) / 2)  # Asegurar índice positivo
        hough_space[theta_deg, rho_index] = 255

# Mostrar la imagen con líneas detectadas y el espacio de Hough
plt.figure(figsize=(10, 5))

# Imagen original con líneas
plt.subplot(1, 2, 1)
plt.title("Líneas detectadas")
plt.imshow(cv2.cvtColor(dibujada, cv2.COLOR_BGR2RGB))
plt.axis("off")

# Espacio de Hough
plt.subplot(1, 2, 2)
plt.title("Espacio de Hough")
plt.imshow(hough_space, cmap="gray", aspect="auto", extent=[-hough_space.shape[1]/2, hough_space.shape[1]/2, 180, 0])
plt.xlabel("\u03C1 (Distancia)")
plt.ylabel("\u03B8 (Ángulo en grados)")

plt.tight_layout()
plt.show()