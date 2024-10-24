import cv2
import numpy as np

# Load pre-trained gender classification model
gender_net = cv2.dnn.readNetFromCaffe(
    prototxt='deploy_gender.prototxt',
    caffeModel='gender_net.caffemodel'
)

# List of gender labels
gender_labels = ['Male', 'Female']

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame from the camera
    ret, frame = cap.read()

    # Perform face detection using Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]

        # Preprocess face ROI for gender classification
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746))

        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender_index = np.argmax(gender_preds)
        gender = gender_labels[gender_index]

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Write gender text
        text = f'{gender}'
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame in a window
    cv2.imshow("Gender Classification", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
