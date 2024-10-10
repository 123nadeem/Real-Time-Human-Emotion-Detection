import cv2
import numpy as np
from ultralytics import YOLO

# Load the trained YOLOv8 classification model
model = YOLO('runs/classify/train/weights/best.pt')  # Ensure 'best.pt' is the saved model after training

# Define emotion labels (adjust according to your dataset)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the pre-trained face detection model (Haar Cascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    # Loop through all detected faces
    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI)
        face_roi = frame[y:y+h, x:x+w]

        # Resize the face to the size required by the model (48x48)
        resized_face = cv2.resize(face_roi, (48, 48))

        # Use YOLOv8's built-in preprocessing, pass the face ROI directly
        results = model.predict(resized_face, save=False)

        # Ensure results are not empty
        if len(results) > 0:
            # Extract the predicted probabilities from the 'probs' object
            probs = results[0].probs

            # Get the top predicted emotion index and confidence
            max_prob_idx = probs.top1  # Index of the class with the highest probability
            confidence = probs.top1conf.item()  # Confidence score for the top class

            # Get the predicted emotion label
            emotion = emotion_labels[max_prob_idx]

            # Draw a rectangle around the face and display the emotion label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            text = f"{emotion} ({confidence:.2f})"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Show the final output frame
    cv2.imshow('Emotion Detection', frame)

    # Check if 'q' is pressed or if the window is closed manually
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Check if the window was closed manually
    if cv2.getWindowProperty('Emotion Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
