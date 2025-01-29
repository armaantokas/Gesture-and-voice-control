import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

gestures = ["up", "down", "left", "right", "forward", "backward"]
output_dir = "gesture_data"
os.makedirs(output_dir, exist_ok=True)

def collect_gesture_data(gesture_name):
    print(f"Collecting data for gesture: {gesture_name}")
    data = []
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                data.append(landmarks)

        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to finish", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Gesture Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    np.save(os.path.join(output_dir, f"{gesture_name}.npy"), np.array(data))
    print(f"Data saved for gesture: {gesture_name}")

for gesture in gestures:
    collect_gesture_data(gesture)

print("Gesture data collection completed!")
