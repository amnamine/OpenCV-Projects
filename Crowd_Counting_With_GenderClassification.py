import cv2
import numpy as np

# Load the pre-trained Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained gender classification model
gender_net = cv2.dnn.readNetFromCaffe(
    prototxt='deploy_gender.prototxt',
    caffeModel='gender_net.caffemodel'
)

# List of gender labels
gender_labels = ['Male', 'Female']

# Open a connection to the camera. 0 indicates the default camera, you can change it to another index if needed.
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Initialize counters for males, females, and total faces
    num_male = 0
    num_female = 0

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

        # Draw rectangle around the face and write gender text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        text = f'{gender}'
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Increment the respective gender counter
        if gender == 'Male':
            num_male += 1
        elif gender == 'Female':
            num_female += 1

    # Calculate total faces detected
    total_faces = len(faces)

    # Display the counts of males, females, and total faces detected
    cv2.putText(frame, f'Male: {num_male}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Female: {num_female}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Total Faces: {total_faces}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
video_capture.release()
cv2.destroyAllWindows()
