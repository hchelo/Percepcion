import tensorflow as tf
import cv2
import numpy as np

# Ruta al modelo y al video
PATH_TO_CKPT = "frozen_inference_graph_V2.pb"
VIDEO_PATH = "carros.mp4"

# Función para cargar el modelo
def load_model():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")
    return detection_graph

# Cargar el modelo
print("Cargando modelo...")
detection_graph = load_model()
print("Modelo cargado.")

# Procesar video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"No se puede abrir el video: {VIDEO_PATH}")
    exit()

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
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        label = f"Car {scores[0][i]:.2f}"
                        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Mostrar cuadro procesado
            cv2.imshow("Detección de Carros", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
