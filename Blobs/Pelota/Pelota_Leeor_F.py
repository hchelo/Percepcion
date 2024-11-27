import cv2
from detector_pelota import detector_pelota

# Ruta del video
# Ruta del video
video_path = 'futbol.mp4'

# Llamar a la función detector_pelota para obtener las coordenadas de la pelota
detector_pelota(video_path)

# Usar las coordenadas de la pelota en el programa principal
