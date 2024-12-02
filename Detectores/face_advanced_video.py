from mtcnn import MTCNN
import cv2
import os

# Inicializar el detector MTCNN
detector = MTCNN()

# Ruta del video
video_path = "dragon.mp4"
output_folder = "output_faces"  # Carpeta para guardar los rostros detectados

# Crear la carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Cargar el video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: No se puede abrir el video {video_path}")
    exit()

frame_count = 0  # Contador de fotogramas
face_count = 0   # Contador de caras detectadas

# Procesar cada fotograma del video
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
    for idx, result in enumerate(results):
        x, y, width, height = result['box']
        # Ajustar las coordenadas para asegurarse de que estén dentro de los límites
        x, y = max(0, x), max(0, y)
        width, height = max(0, width), max(0, height)

        # Extraer la región de la cara detectada
        #face = frame[y:y + height, x:x + width]

        # Guardar la imagen de la cara detectada
        #face_filename = os.path.join(output_folder, f"frame{frame_count}_face{idx + 1}.png")
        #cv2.imwrite(face_filename, face)
        #print(f"Rostro guardado en: {face_filename}")

        # Dibujar el rectángulo en el fotograma original
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        face_count += 1

    # Mostrar el fotograma con los rostros detectados
    cv2.imshow("Detected Faces", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Procesamiento completado. Total de fotogramas procesados: {frame_count}")
print(f"Total de rostros detectados: {face_count}")

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
