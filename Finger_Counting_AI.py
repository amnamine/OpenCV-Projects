import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize finger states
thumb_up = 0
index_up = 0
middle_up = 0
ring_up = 0
little_up = 0
total = 0
dis = 180


while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the hand landmarks
    results = hands.process(rgb_frame)
    hand_up_list = []
    if results.multi_hand_landmarks:
        # Iterate through each hand landmark
        registered = False
        hand_up_list = [0] * len(results.multi_hand_landmarks)
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmark points
            landmark_points = []
            for point in hand_landmarks.landmark:
                x, y = int(point.x * frame.shape[1]), int(point.y * frame.shape[0])
                landmark_points.append((x, y))

            # Define finger landmark indices
            wrist = landmark_points[0]
            thumb_tip = landmark_points[4]
            index_tip = landmark_points[8]
            middle_tip = landmark_points[12]
            ring_tip = landmark_points[16]
            little_tip = landmark_points[20]
            
            # Calculate distances between fingers
            thumb_to_wrist = cv2.norm(np.array(thumb_tip) - np.array(wrist))
            index_to_wrist = cv2.norm(np.array(index_tip) - np.array(wrist))
            middle_to_wrist = cv2.norm(np.array(middle_tip) - np.array(wrist))
            ring_to_wrist = cv2.norm(np.array(ring_tip) - np.array(wrist))
            little_to_wrist = cv2.norm(np.array(little_tip) - np.array(wrist))
            
            # Define state for each finger on distance from wrist point
            hand_total = thumb_up + index_up + middle_up + ring_up + little_up
            if thumb_to_wrist > dis-30:
                thumb_up = 1
            else:
                thumb_up = 0

            if index_to_wrist > dis:
                index_up = 1
            else:
                index_up = 0

            if middle_to_wrist > dis:
                middle_up = 1
            else:
                middle_up = 0

            if ring_to_wrist > dis:
                ring_up = 1
            else:
                ring_up = 0

            if little_to_wrist > dis:
                little_up = 1
            else:
                little_up = 0


            hand_up_list.append(hand_total)
            hand_up_list.pop(0)

            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    # Display number of fingers that r up
    countup = 0
    for i in hand_up_list:
        countup += i
    total = countup
    cv2.putText(frame, f"number of fingers up: {total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)
    
    # Exit loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
