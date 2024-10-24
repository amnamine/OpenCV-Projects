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

# Initialize counters
press_count = 0
press_in_progress = False

def detect_shoulder_press(landmarks):
    # Get coordinates for relevant landmarks: shoulder and elbow
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    
    # Conditions to detect a shoulder press:
    # Elbows should be above the shoulders at the top of the press, and below at the bottom.
    left_shoulder_height = left_shoulder.y
    right_shoulder_height = right_shoulder.y
    left_elbow_height = left_elbow.y
    right_elbow_height = right_elbow.y

    # Detect if both elbows are above the shoulders (press up) or below (press down)
    if left_elbow_height < left_shoulder_height and right_elbow_height < right_shoulder_height:
        return "Press Up"
    elif left_elbow_height > left_shoulder_height and right_elbow_height > right_shoulder_height:
        return "Press Down"
    return None

# Create a named window and set it to full screen
cv2.namedWindow("Shoulder Press Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Shoulder Press Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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

        # Classify whether the person is in the "press up" or "press down" position
        position = detect_shoulder_press(landmarks)
        
        # Handle counting logic
        if position == "Press Up" and not press_in_progress:
            press_count += 1
            press_in_progress = True
        elif position == "Press Down":
            press_in_progress = False

        # Display the count on the frame
        cv2.putText(frame, f"Shoulder Press Count: {press_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the video frame in full screen
    cv2.imshow("Shoulder Press Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
