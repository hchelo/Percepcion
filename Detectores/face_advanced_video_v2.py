from mtcnn import MTCNN
import cv2
import os

# Inicializar el detector MTCNN
detector = MTCNN()

# Ruta del video
video_path = "Dragon.mp4"
output_folder = "output_faces"  # Carpeta para guardar los rostros detectados

# Crear la carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Cargar el video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: No se puede abrir el video {video_path}")
    exit()

frame_count = 0  # Contador de fotogramas procesados
face_count = 0   # Contador de caras detectadas
frame_skip = 5  # Saltar de 50 en 50 fotogramas

# Procesar el video
while True:
    # Saltar fotogramas
    for _ in range(frame_skip):
        ret = cap.grab()  # Leer un fotograma sin procesarlo
        if not ret:
            print("Fin del video o error al leer un fotograma.")
            break

    # Leer el siguiente fotograma para procesar
    ret, frame = cap.read()
    if not ret:
        print("Fin del video o error al leer un fotograma.")
        break

    frame_count += frame_skip
    print(f"Procesando fotograma {frame_count}...")

    # Convertir el fotograma a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar caras
    results = detector.detect_faces(rgb_frame)

    # Dibujar rectángulos alrededor de las caras detectadas y guardar las imágenes
    for idx, result in enumerate(results):
        x, y, width, height = result['box']
        x, y = max(0, x), max(0, y)
        width, height = max(0, width), max(0, height)

        # Extraer y guardar la región de la cara detectada
        face = frame[y:y + height, x:x + width]
        face_filename = os.path.join(output_folder, f"frame{frame_count}_face{idx + 1}.png")
        #cv2.imwrite(face_filename, face)
        print(f"Rostro guardado en: {face_filename}")

        # Dibujar el rectángulo en el fotograma original
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        face_count += 1

    # Mostrar el fotograma con las caras detectadas
    cv2.imshow("Detected Faces", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Procesamiento completado. Total de fotogramas procesados: {frame_count}")
print(f"Total de rostros detectados: {face_count}")

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
