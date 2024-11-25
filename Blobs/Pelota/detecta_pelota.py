import cv2
import numpy as np
import threading
import queue

# Función para detectar la pelota y enviar el valor de x a través de una cola
def detecta_pelota(video_path, queue):
    # Iniciar la captura de video
    cap = cv2.VideoCapture(video_path)

    while True:
        # Leer el siguiente cuadro del video
        ret, frame = cap.read()
        
        if not ret:
            break  # Si no hay más cuadros, salir del bucle

        # Obtener la altura y ancho del cuadro
        height, width, _ = frame.shape
        
        # Crear una máscara del mismo tamaño que el cuadro original
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Recorrer todos los píxeles del cuadro y cambiar su color
        # Filtrar el color de la pelota de manera más eficiente
        red_mask = np.logical_and.reduce((frame[:, :, 2] <= 190, frame[:, :, 2] >= 100, 
                                        frame[:, :, 1] <= 201, frame[:, :, 1] >= 158, 
                                        frame[:, :, 0] <= 82, frame[:, :, 0] >= 34))

        # Crear una máscara binaria con los píxeles correspondientes a la pelota
        #mask[:, :][red_mask] = 255
        mask[150:, :][red_mask[150:, :]] = 255

        # Aplicar operaciones morfológicas (cierre y dilatación) a la máscara
        kernel = np.ones((35, 35), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Encontrar los contornos en la máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Encontrar el contorno más grande
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(max_contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(frame, center, radius, (0, 0, 255), 2)
            cv2.putText(frame, "Pelota", (int(x-radius), int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # Enviar el valor de x a través de la cola
            queue.put(center)
        
        # Mostrar el cuadro resultante
        cv2.imshow('Video', frame)
        
        # Detener la ejecución del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Liberar la captura de video y cerrar todas las ventanas
    cap.release()
    cv2.destroyAllWindows()

# Función que ejecuta la detección de la pelota en un hilo separado
def run_detection(video_path, queue):
    threading.Thread(target=detecta_pelota, args=(video_path, queue), daemon=True).start()

# Programa principal
if __name__ == "__main__":
    # Crear una cola para comunicarse entre hilos
    q = queue.Queue()
    # Ruta del video
    video_path = 'futbol.mp4'
    # Iniciar la detección de la pelota en un hilo separado
    run_detection(video_path, q)
    # Obtener y mostrar el valor de x en tiempo real
    while True:
        try:
            center = q.get_nowait()
            #print("Valor de x:", x)
            x = center[0]
            y = center[1]
            # Ahora puedes usar x_coordinate y y_coordinate como necesites
            print("Coordenada x:", x)
            print("Coordenada y:", y)
        except queue.Empty:
            pass
