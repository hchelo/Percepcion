import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Configuración del modelo ResNet
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('resnet50_model.pth', map_location=torch.device('cpu')))
model.eval()

# Transformación para preprocesar cada slice
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Ruta del video
video_path = 'carros.mp4'

# Cargar el video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error al abrir el video.")
else:
    # Configuración inicial
    skip_frames = 25
    window_size = (160, 160)
    stride = 30
    resize_dim = (300, 300)

    frame_count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = resize_dim[0]
    frame_height = resize_dim[1]

    # Configuración para grabar la salida
    output_path = 'output_carros.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        # Leer el frame
        ret, frame = cap.read()
        if not ret:
            break

        # Saltar frames no deseados
        if frame_count % skip_frames != 0:
            frame_count += 1
            continue

        # Redimensionar el frame
        frame_resized = cv2.resize(frame, resize_dim)
        h, w, _ = frame_resized.shape

        # Mejor predicción en el frame actual
        best_prediction = None
        best_confidence = -1

        # Procesar con ventana deslizante
        for y in range(0, h - window_size[0] + 1, stride):
            for x in range(0, w - window_size[1] + 1, stride):
                # Extraer el slice
                window_slice = frame_resized[y:y + window_size[0], x:x + window_size[1]]

                # Preprocesar el slice
                pil_image = Image.fromarray(window_slice)
                image_tensor = transform(pil_image).unsqueeze(0)

                # Predicción
                with torch.no_grad():
                    outputs = model(image_tensor)
                    _, predicted = torch.max(outputs, 1)
                    confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()

                # Actualizar si la confianza es mayor
                if confidence > 0.97 and confidence > best_confidence:
                    best_confidence = confidence
                    best_prediction = (x, y, x + window_size[1], y + window_size[0], predicted.item())

        # Dibujar la mejor predicción
        if best_prediction:
            x_start, y_start, x_end, y_end, predicted_class = best_prediction
            rect_color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)
            label = f'{"Beatle" if predicted_class == 0 else "Willys"}: {best_confidence:.2f}%'

            cv2.rectangle(frame_resized, (x_start, y_start), (x_end, y_end), rect_color, 2)
            cv2.putText(frame_resized, label, (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 2)

        # Mostrar el frame
        cv2.imshow('Deteccion', frame_resized)

        # Guardar el frame procesado en el archivo de salida
        out.write(frame_resized)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # Liberar recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()
