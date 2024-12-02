from mtcnn import MTCNN
import cv2
import os

# Inicializar el detector MTCNN
detector = MTCNN()

# Ruta del video de entrada
input_video_path = "Dragon.mp4"
output_video_path = "output_video.mp4"  # Ruta del video de salida

# Cargar el video
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Error: No se puede abrir el video {input_video_path}")
    exit()

# Obtener propiedades del video de entrada
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Inicializar el VideoWriter para guardar el video de salida
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para el formato MP4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0  # Contador de fotogramas procesados

# Procesar el video
while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin del video o error al leer un fotograma.")
        break

    frame_count += 1
    #print(f"Procesando fotograma {frame_count}...")

    # Convertir el fotograma a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar caras
    results = detector.detect_faces(rgb_frame)

    # Dibujar rectángulos alrededor de las caras detectadas
    for result in results:
        x, y, width, height = result['box']
        x, y = max(0, x), max(0, y)
        width, height = max(0, width), max(0, height)

        # Dibujar el rectángulo en el fotograma
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Escribir el fotograma procesado en el video de salida
    out.write(frame)

    # Mostrar el fotograma con las detecciones
    cv2.imshow("Detected Faces", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Procesamiento completado. Total de fotogramas procesados: {frame_count}")

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
