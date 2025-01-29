import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import speech_recognition as sr

gesture_model = load_model("gesture_classification_model.h5")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

gesture_labels = ["up", "down", "left", "right", "forward", "backward"]

recognizer = sr.Recognizer()

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 

frame_counter = 0
prediction_interval = 1

last_predicted_label_1 = "No Gesture"
last_predicted_label_2 = "No Gesture"

def listen_for_voice_command():
    with sr.Microphone() as source:
        print("Listening for voice commands...")
        try:
            audio = recognizer.listen(source, timeout=8)
            command = recognizer.recognize_google(audio)
            print(f"Voice Command: {command}")
            return command.lower()
        except sr.UnknownValueError:
            print("Could not understand the voice command.")
            return None
        except sr.RequestError:
            print("Could not request results.")
            return None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

            if frame_counter % prediction_interval == 0:
                prediction = gesture_model.predict(np.array([landmarks]), verbose=0)
                predicted_label = gesture_labels[np.argmax(prediction)]
                
                if i == 0:
                    last_predicted_label_1 = predicted_label  
                elif i == 1:
                    last_predicted_label_2 = predicted_label  
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"D1: {last_predicted_label_1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if len(results.multi_hand_landmarks) > 1:
            cv2.putText(frame, f"D2: {last_predicted_label_2}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    frame_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('v'):
        command = listen_for_voice_command()

        if command:
            if "up" in command:
                print("Voice command detected: UP")
            elif "down" in command:
                print("Voice command detected: DOWN")
            elif "left" in command:
                print("Voice command detected: LEFT")
            elif "right" in command:
                print("Voice command detected: RIGHT")
            elif "forward" in command:
                print("Voice command detected: FORWARD")
            elif "backward" in command:
                print("Voice command detected: BACKWARD")
            elif "exit" in command:
                print("Exiting gesture recognition.")
                break

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
