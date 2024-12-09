import tensorflow as tf
import cv2
import numpy as np
from torchvision import models, transforms
import torch
import torch.nn as nn
from PIL import Image

# Ruta al modelo y al video
PATH_TO_CKPT = "frozen_inference_graph_V2.pb"
VIDEO_PATH = "carros.mp4"
OUTPUT_PATH = "output_proy_carros.mp4"  # Ruta para el archivo de salida

# Cargar el modelo de clasificación de vehículos (Willys/Beatle)
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Asumiendo que son 2 clases: Willys y Beatle
model.load_state_dict(torch.load('resnet50_model.pth', map_location=torch.device('cpu')))
model.eval()

# Transformación para preprocesar las imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Función para cargar el modelo de detección
def load_model():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")
    return detection_graph

# Cargar el modelo de detección
print("Cargando modelo de detección...")
detection_graph = load_model()
print("Modelo de detección cargado.")

# Procesar video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"No se puede abrir el video: {VIDEO_PATH}")
    exit()

# Obtener las propiedades del video original
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Configurar el escritor de video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

# Crear sesión de detección
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Fin del video

            # Preprocesar el cuadro
            input_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
            detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
            detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
            detection_classes = detection_graph.get_tensor_by_name("detection_classes:0")
            num_detections = detection_graph.get_tensor_by_name("num_detections:0")

            # Realizar detección
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            expanded_frame = np.expand_dims(frame_rgb, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={input_tensor: expanded_frame},
            )

            # Dibujar detecciones en el cuadro
            h, w, _ = frame.shape
            for i in range(int(num[0])):
                if scores[0][i] > 0.5:  # Umbral de confianza
                    class_id = int(classes[0][i])
                    if class_id == 3:  # ID de 'car' en COCO
                        box = boxes[0][i] * [h, w, h, w]
                        y_min, x_min, y_max, x_max = box.astype(int)

                        # Extraer la porción del vehículo (carro)
                        car_image = frame[y_min:y_max, x_min:x_max]
                        car_image_resized = cv2.resize(car_image, (300, 300))  # Redimensionar a 300x300

                        # Preprocesar la imagen para el clasificador
                        pil_image = Image.fromarray(car_image_resized)
                        image_tensor = transform(pil_image).unsqueeze(0)

                        # Realizar la predicción (Willys vs Beatle)
                        with torch.no_grad():
                            outputs = model(image_tensor)
                            _, predicted = torch.max(outputs, 1)
                            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()

                        # Marcar el tipo de vehículo
                        label = "Willys" if predicted.item() == 1 else "Beatle"
                        color = (0, 255, 0) if label == "Willys" else (0, 0, 255)

                        # Dibujar el rectángulo y la etiqueta sobre el cuadro original
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                        cv2.putText(frame, f'{label}: {confidence*100:.2f}%', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Guardar el cuadro procesado en el archivo de salida
            out.write(frame)

            # Mostrar cuadro procesado
            cv2.imshow("Detección de Carros", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
