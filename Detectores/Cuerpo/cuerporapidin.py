import cv2
import numpy as np

# Create our body classifier
body_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
# Initiate video capture for video file, here we are using the video file in which pedestrians would be detected
cap = cv2.VideoCapture('EMI.mp4')

frame_counter = 0

# Loop once video is successfully loaded
while cap.isOpened():
    # Read the current frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # Increment the frame counter
    frame_counter += 1
    
    # Skip every other frame
    if frame_counter % 20 != 0:
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.1, 3)

    # Extract bounding boxes for any bodies identified
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Show the processed frame
    cv2.imshow('Pedestrians', frame)

    # Break the loop if Enter key is pressed
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
