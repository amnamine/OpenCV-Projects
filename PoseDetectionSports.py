import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

def classify_pose(landmarks):
    # Get coordinates of relevant landmarks: shoulders, wrists, and elbows
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # T-Pose detection: wrists should be at approximately the same height as the shoulders
    if (abs(left_wrist.y - left_shoulder.y) < 0.1 and
        abs(right_wrist.y - right_shoulder.y) < 0.1):
        return "T-Pose"
    
    # Hand Raised detection: wrists should be significantly above the shoulders
    elif (left_wrist.y < left_shoulder.y - 0.2 and
          right_wrist.y < right_shoulder.y - 0.2):
        return "Hand Raised"
    
    # Hand on Hip detection: wrist should be near the hip in both x and y coordinates
    elif (abs(left_wrist.y - left_hip.y) < 0.2 and abs(left_wrist.x - left_hip.x) < 0.2 or
          abs(right_wrist.y - right_hip.y) < 0.2 and abs(right_wrist.x - right_hip.x) < 0.2):
        return "Hand on Hip"
    
    # Surrender pose detection: both arms raised above the head with bent elbows
    elif (left_wrist.y < left_shoulder.y - 0.4 and left_elbow.y < left_shoulder.y - 0.1 and
          right_wrist.y < right_shoulder.y - 0.4 and right_elbow.y < right_shoulder.y - 0.1):
        return "Surrender"
    
    # If neither T-Pose, Hand Raised, Hand on Hip, nor Surrender, classify as Normal
    else:
        return "Normal"

# Create a named window and set it to full screen
cv2.namedWindow('Pose Estimation', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Pose Estimation', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the image to RGB as mediapipe uses RGB images
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and find the pose
    result = pose.process(img_rgb)

    # Check if any landmarks are detected
    if result.pose_landmarks:
        # Draw the pose landmarks on the video frame
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get the landmarks as a list
        landmarks = result.pose_landmarks.landmark

        # Classify whether the person is in T-Pose, Hand Raised, Hand on Hip, Surrender, or Normal
        posture = classify_pose(landmarks)
        
        # Display the classification on the frame
        cv2.putText(frame, posture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the video frame in full screen
    cv2.imshow('Pose Estimation', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
