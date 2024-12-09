import tensorflow.compat.v1 as tf  # TensorFlow 1.x en modo compatibilidad
tf.disable_v2_behavior()
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Rutas al modelo y etiquetas
PATH_TO_CKPT = "frozen_inference_graph_V2.pb"
PATH_TO_LABELS = "mscoco_label_map.pbtxt"

# Cargar el modelo en un grafo de TensorFlow
def load_model():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

detection_graph = load_model()

# Función para procesar imágenes
def detect_objects(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Obtener tensores de entrada y salida
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Preparar entrada
            input_image = np.expand_dims(img_rgb, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: input_image}
            )

            # Procesar detecciones
            for i in range(int(num[0])):
                if scores[0][i] > 0.5:  # Umbral de confianza
                    class_id = int(classes[0][i])
                    if class_id == 3:  # ID de 'car' en COCO
                        box = boxes[0][i] * [h, w, h, w]
                        y_min, x_min, y_max, x_max = box.astype('int')

                        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        label = f"Car {scores[0][i]:.2f}"
                        cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar resultados
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Ejecutar detección
detect_objects("autos.jpg")
