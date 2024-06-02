import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to classify hand gestures.
def classify_hand_gesture(landmarks):
    # List of fingers (thumb, index, middle, ring, pinky) based on landmark points.
    finger_tips = [4, 8, 12, 16, 20]
    finger_dips = [3, 7, 11, 15, 19]

    finger_states = []
    for tip, dip in zip(finger_tips, finger_dips):
        if landmarks[tip].y < landmarks[dip].y:
            finger_states.append(1)  # Finger is up
        else:
            finger_states.append(0)  # Finger is down

    if finger_states == [0, 0, 0, 0, 0]:
        return "Fist"
    elif finger_states == [1, 1, 1, 1, 1]:
        return "Open Palm"
    elif finger_states == [1, 0, 0, 0, 0]:
        return "Thumbs Up"
    else:
        return "Unknown Gesture"

# Start capturing video from the webcam.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands.
    result = hands.process(rgb_frame)

    # Draw hand landmarks on the frame and classify gestures.
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )

            # Convert landmarks to a list of (x, y) coordinates.
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            
            # Classify and print the gesture.
            gesture = classify_hand_gesture(hand_landmarks.landmark)
            print("Detected Gesture:", gesture)

    # Display the frame.
    cv2.imshow('Hand Gesture Detection', frame)

    # Exit on pressing 'q'.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
