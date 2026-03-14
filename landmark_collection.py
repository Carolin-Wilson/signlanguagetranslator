import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# Ask for label
label = input("Enter letter label (A/B/C/D/E): ").upper()

# Create dataset folder if not exists
if not os.path.exists("dataset"):
    os.makedirs("dataset")

file_path = f"dataset/{label}.csv"

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

count = 0

print("Collecting landmarks...")

with open(file_path, mode='a', newline='') as f:
    writer = csv.writer(f)

    while count < 200:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y

                landmarks = []

                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x - wrist_x)
                    landmarks.append(lm.y - wrist_y)

                writer.writerow(landmarks)
                count += 1

        cv2.putText(frame, f"Samples: {count}/200", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Landmark Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print(f"Finished collecting for {label}")