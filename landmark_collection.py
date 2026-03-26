import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
import argparse

# ── CLI argument support ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Collect hand landmark data for sign language gestures.")
parser.add_argument("--label", type=str, default=None,
                    help="Label to collect (e.g. A, B, HELLO, YES). Skips interactive prompt.")
parser.add_argument("--samples", type=int, default=200,
                    help="Number of samples to collect (default: 200).")
args = parser.parse_args()

# ── Label input ───────────────────────────────────────────────────────────────
if args.label:
    label = args.label.upper()
else:
    print("\n" + "="*50)
    print("  Sign Language Landmark Collector")
    print("="*50)
    print("  Enter a LETTER (A-Z) or a WORD (e.g. HELLO, YES, NO)")
    label = input("  Label: ").strip().upper()

if not label:
    print("No label provided. Exiting.")
    exit()

TARGET_SAMPLES = args.samples

# ── Dataset setup ─────────────────────────────────────────────────────────────
os.makedirs("dataset", exist_ok=True)
file_path = f"dataset/{label}.csv"

# Count existing samples
existing_count = 0
if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        existing_count = sum(1 for _ in f)

remaining = max(0, TARGET_SAMPLES - existing_count)

print(f"\n  Label       : {label}")
print(f"  Target      : {TARGET_SAMPLES} samples")
print(f"  Existing    : {existing_count} samples")
print(f"  To collect  : {remaining} samples")

if remaining == 0:
    print(f"\n  ✅  Already have {existing_count} samples for '{label}'. Nothing to collect.")
    exit()

input("\n  Press ENTER to start the countdown...")

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

# ── Countdown ─────────────────────────────────────────────────────────────────
COUNTDOWN = 3
for i in range(COUNTDOWN, 0, -1):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"Get ready: {i}", (w//2 - 130, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 100), 4)
    cv2.putText(frame, f"Label: {label}", (w//2 - 80, h//2 + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
    cv2.imshow("Landmark Collection", frame)
    cv2.waitKey(1000)

# ── Collection loop ───────────────────────────────────────────────────────────
count = 0

with open(file_path, mode='a', newline='') as f:
    writer = csv.writer(f)

    while count < remaining:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        landmark_written = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x - wrist_x)
                    landmarks.append(lm.y - wrist_y)

                writer.writerow(landmarks)
                count += 1
                landmark_written = True

        # ── UI ────────────────────────────────────────────────────────────────
        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 90), (30, 30, 30), -1)
        cv2.putText(frame, f"Label: {label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 2)
        cv2.putText(frame, f"Sample {count}/{remaining}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        # Progress bar
        bar_x, bar_y, bar_w, bar_h = 20, h - 45, w - 40, 20
        progress = int(bar_w * count / remaining)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress, bar_y + bar_h), (0, 220, 100), -1)
        pct = int(100 * count / remaining)
        cv2.putText(frame, f"{pct}%", (bar_x + bar_w // 2 - 15, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Hand status indicator
        status_color = (0, 220, 100) if landmark_written else (0, 50, 200)
        status_text = "Capturing" if landmark_written else "No hand detected"
        cv2.circle(frame, (w - 30, 30), 10, status_color, -1)
        cv2.putText(frame, status_text, (w - 200, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)

        cv2.imshow("Landmark Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nCollection interrupted by user.")
            break

cap.release()
cv2.destroyAllWindows()

total = existing_count + count
print(f"\n  ✅  Done! Collected {count} new samples for '{label}'.")
print(f"      Total for '{label}': {total} samples.")
print(f"\n  Tip: Run `python train_model.py` to retrain the model with the new data.\n")