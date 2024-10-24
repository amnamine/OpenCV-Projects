import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('age.h5')

# Function to preprocess the image for prediction
def preprocess_image(image):
    image = cv2.resize(image, (64, 64))  # Resize to the expected input size
    image = image.reshape((1, 64, 64, 1))  # Reshape for the model
    return image / 255.0  # Normalize the image

# Function to convert predicted age distribution to age group
def get_age(distr):
    distr = distr * 4
    if distr >= 0.65 and distr <= 1.4:
        return "0-18"
    elif distr >= 1.65 and distr <= 2.4:
        return "19-30"
    elif distr >= 2.65 and distr <= 3.4:
        return "31-80"
    elif distr >= 3.65 and distr <= 4.4:
        return "80+"
    return "Unknown"

# Function to get gender from predicted probability
def get_gender(prob):
    return "Male" if prob < 0.5 else "Female"

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Preprocess the frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    processed_frame = preprocess_image(gray_frame)  # Preprocess for model

    # Make predictions
    predictions = model.predict(processed_frame)
    age_group = get_age(predictions[0][0])  # Get age group
    gender = get_gender(predictions[1][0])   # Get gender

    # Display results on the frame
    cv2.putText(frame, f'Age: {age_group}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'Gender: {gender}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Age and Gender Estimation', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
