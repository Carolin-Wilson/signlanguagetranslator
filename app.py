import cv2
import numpy as np
import mediapipe as mp
import pickle
import time
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model/sign_language_model.h5")

# Load label encoder
with open("model/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Camera
cap = cv2.VideoCapture(0)

# Sentence logic variables
sentence = ""
current_letter = ""
letter_start_time = 0
letter_confirmed = False

confidence_threshold = 0.85
hold_time_required = 0.7
no_hand_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    predicted_letter = "-"
    confidence = 0.0

    if results.multi_hand_landmarks:
        no_hand_frames = 0

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x - wrist_x)
                landmarks.append(lm.y - wrist_y)

            landmarks = np.array(landmarks).reshape(1, -1)

            prediction = model.predict(landmarks, verbose=0)
            confidence = np.max(prediction)
            predicted_letter = le.inverse_transform([np.argmax(prediction)])[0]

            current_time = time.time()

            if confidence > confidence_threshold:

                # If new letter detected → reset timer and confirmation
                if predicted_letter != current_letter:
                    current_letter = predicted_letter
                    letter_start_time = current_time
                    letter_confirmed = False

                # If same letter held long enough AND not already confirmed
                elif (current_time - letter_start_time) > hold_time_required and not letter_confirmed:
                    sentence += current_letter
                    letter_confirmed = True

    else:
        no_hand_frames += 1
        current_letter = ""
        letter_confirmed = False

        if no_hand_frames > 40:
            if len(sentence) > 0 and sentence[-1] != " ":
                sentence += " "
    # ---------- UI SECTION ----------

    height, width, _ = frame.shape

    # Top Panel Background
    cv2.rectangle(frame, (0, 0), (width, 100), (30, 30, 30), -1)

    # Bottom Panel Background
    cv2.rectangle(frame, (0, height - 120), (width, height), (20, 20, 20), -1)

    # Display Letter
    cv2.putText(frame, f"Letter: {predicted_letter}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    # Display Confidence
    cv2.putText(frame, f"Confidence: {confidence:.2f}",
                (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2)

    # Display Sentence
    cv2.putText(frame, f"Sentence: {sentence}",
                (30, height - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)

    cv2.imshow("Sign Language Translator", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press Q to exit
    if key == ord('q'):
        break

    # Press C to clear sentence
    if key == ord('c'):
        sentence = ""
        last_added_letter = ""

    # Press Backspace to delete last letter
    if key == 8:
        sentence = sentence[:-1]

cap.release()
cv2.destroyAllWindows()