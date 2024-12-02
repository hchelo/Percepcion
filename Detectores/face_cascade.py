import cv2

def draw_found_faces(detected, image, color: tuple):
    for (x, y, width, height) in detected:
        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            color,
            thickness=2
        )

def filter_overlapping_detections(profiles, faces, overlap_threshold=0.5):
    def compute_overlap(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        overlap_area = x_overlap * y_overlap
        area1 = w1 * h1
        area2 = w2 * h2
        return overlap_area / min(area1, area2)

    return [
        profile for profile in profiles 
        if not any(compute_overlap(profile, face) > overlap_threshold for face in faces)
    ]

path_to_image = "face_rot/familia.png"
original_image = cv2.imread(path_to_image)

if original_image is not None:
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Cargar clasificadores Haar
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

    if face_cascade.empty() or profile_cascade.empty():
        print("Error al cargar los clasificadores Haar. Verifica la ruta y archivos.")
        exit()

    # Detectar rostros y perfiles
    detected_faces = face_cascade.detectMultiScale(image=image, scaleFactor=1.1, minNeighbors=5)
    detected_profiles = profile_cascade.detectMultiScale(image=image, scaleFactor=1.1, minNeighbors=5)

    # Filtrar perfiles que ya son rostros
    profiles_not_faces = filter_overlapping_detections(detected_profiles, detected_faces)

    # Dibujar detecciones
    draw_found_faces(detected_faces, original_image, (0, 255, 0))  # Verde para rostros
    draw_found_faces(profiles_not_faces, original_image, (0, 0, 255))  # Rojo para perfiles

    # Mostrar resultados
    cv2.imshow(f'Detected Faces in {path_to_image}', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f'An error occurred while trying to load {path_to_image}')
