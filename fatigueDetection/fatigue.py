import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe face mesh for landmark detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# EAR and MAR thresholds
EAR_THRESHOLD = 0.4  # Threshold for eye closure
MAR_THRESHOLD = 1.2   # Threshold for yawning
CONSEC_FRAMES_SLEEP = 15  # Number of consecutive frames to consider "Sleeping"
CONSEC_FRAMES_YAWN = 5    # Number of consecutive frames to consider "Yawning"

# Variables to count consecutive frames for each state
ear_counter = 0
mar_counter = 0

def compute_ear(eye):
    # Calculate EAR based on eye landmarks
    return (np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])) / (2.0 * np.linalg.norm(eye[0] - eye[3]))

def compute_mar(mouth):
    # Calculate MAR based on mouth landmarks
    return np.linalg.norm(mouth[3] - mouth[9]) / np.linalg.norm(mouth[0] - mouth[6])

# Main loop to capture video frames
cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Convert to RGB for Mediapipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)
    
    state = "Active"  # Default state

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Extract landmark points for eyes and mouth
            left_eye = np.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in [362, 385, 387, 263, 373, 380]])
            right_eye = np.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in [33, 160, 158, 133, 153, 144]])
            mouth = np.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in [78, 81, 13, 311, 308, 402, 317, 14, 87, 178]])

            # Calculate EAR and MAR
            left_ear = compute_ear(left_eye)
            right_ear = compute_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0
            mar = compute_mar(mouth)

            # Check EAR threshold for sleeping state
            if ear < EAR_THRESHOLD:
                ear_counter += 1
            else:
                ear_counter = 0

            # Check MAR threshold for yawning state
            if mar > MAR_THRESHOLD:
                mar_counter += 1
            else:
                mar_counter = 0

            # Determine state based on counters
            if ear_counter >= CONSEC_FRAMES_SLEEP:
                state = "Sleeping"
            elif mar_counter >= CONSEC_FRAMES_YAWN:
                state = "Yawning"
            else:
                state = "Active"

            # Display the current state on the frame
            cv2.putText(frame, f"State: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if state == "Active" else (0, 0, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


    # Show the frame with the state displayed
    cv2.imshow("Fatigue Detection", frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
