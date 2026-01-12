import pickle
import cv2
import mediapipe as mp
import numpy as np
import os

# -----------------------------
# Load the trained model
# -----------------------------
model_path = r'C:\Users\MSI\Downloads\Sign-Language-And-Gesture-Detector-main\Sign-Language-And-Gesture-Detector-main\model.p'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

with open(model_path, 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']

# -----------------------------
# Initialize webcam
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check if it's being used by another application.")

# -----------------------------
# Initialize MediaPipe Hands
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,           # real-time tracking
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Labels dictionary
# -----------------------------
labels_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
               10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
               19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
               28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'}

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract coordinates
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            x1 = max(int(min(x_) * W) - 10, 0)
            y1 = max(int(min(y_) * H) - 10, 0)
            x2 = min(int(max(x_) * W) + 10, W)
            y2 = min(int(max(y_) * H) + 10, H)

            # Prepare feature vector
            data_aux = []
            for i in range(len(x_)):
                data_aux.append(x_[i])
                data_aux.append(y_[i])
            while len(data_aux) < 42:  # pad if needed
                data_aux.append(0.0)
            data_aux = np.array(data_aux).reshape(1, -1)

            # Predict character
            prediction = model.predict(data_aux)
            predicted_character = labels_dict[int(prediction[0])]

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display
    cv2.imshow('ASL Recognition', frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
