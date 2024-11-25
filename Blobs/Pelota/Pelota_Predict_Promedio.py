import cv2
import numpy as np

def detecta_pelota(video_path):
    cap = cv2.VideoCapture(video_path)

    # Variables para almacenar las coordenadas anteriores
    prev_x, prev_y = None, None

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break  

        height, width, _ = frame.shape
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        red_mask = np.logical_and.reduce((frame[:, :, 2] <= 190, frame[:, :, 2] >= 100, 
                                        frame[:, :, 1] <= 201, frame[:, :, 1] >= 158, 
                                        frame[:, :, 0] <= 82, frame[:, :, 0] >= 34))

        mask[150:, :][red_mask[150:, :]] = 255

        kernel = np.ones((35, 35), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(max_contour)
            center = (int(x), int(y))  # Coordenadas del centro de la pelota
            
            # Dibujar la pelota detectada
            cv2.circle(frame, center, int(radius), (0, 0, 255), 2)
            cv2.putText(frame, "Pelota", (int(x-radius), int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Si tenemos coordenadas anteriores, calcular el desplazamiento
            if prev_x is not None and prev_y is not None:
                # Calcular el desplazamiento entre los fotogramas consecutivos
                dx = x - prev_x
                dy = y - prev_y
                
                # Calcular las posiciones predichas para varios pasos futuros (10, 20 y 30)
                for step in [10, 20, 30]:
                    predicted_x = int(x + dx * step)
                    predicted_y = int(y + dy * step)
                    
                    # Dibujar un círculo en la posición predicha
                    cv2.circle(frame, (predicted_x, predicted_y), int(radius), (0, 255, 0), 2)
            
            # Actualizar las coordenadas anteriores
            prev_x, prev_y = x, y
        
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'futbol.mp4'
    detecta_pelota(video_path)
