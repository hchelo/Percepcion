from mtcnn import MTCNN
import cv2

# Cargar imagen
image = cv2.imread("face_rot/familia.png")
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Inicializar MTCNN
detector = MTCNN()

# Detectar caras
results = detector.detect_faces(rgb_image)

# Dibujar rectángulos
for result in results:
    x, y, width, height = result['box']
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
#print(results)
# Mostrar resultados
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
