"""
collect_word_samples.py
-----------------------
Record your own gesture samples to improve word recognition accuracy.

Controls:
  SPACE  - start/confirm recording a sample
  S      - skip current word
  Q      - quit and retrain

Usage:
  python collect_word_samples.py                  # all 38 words
  python collect_word_samples.py HELLO YES NO     # specific words only
"""

import os
import sys
import cv2
import numpy as np
import time
import mediapipe as mp

# ── Config ─────────────────────────────────────────────────────────────────────
SEQUENCE_DATASET = "sequence_dataset"
SEQ_LEN          = 30    # frames to capture per sample
SAMPLES_PER_WORD = 12    # how many recordings to make per word
COUNTDOWN_SEC    = 2     # seconds to prepare after pressing SPACE

# ── MediaPipe ──────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.6, min_tracking_confidence=0.6,
)

# ── Determine word list ────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    WORDS = [w.upper() for w in sys.argv[1:]]
else:
    WORDS = sorted([
        d for d in os.listdir(SEQUENCE_DATASET)
        if os.path.isdir(os.path.join(SEQUENCE_DATASET, d))
    ])

# ── Helpers ────────────────────────────────────────────────────────────────────
def extract_landmarks(hand_landmarks):
    wrist_x = hand_landmarks.landmark[0].x
    wrist_y = hand_landmarks.landmark[0].y
    feats = []
    for lm in hand_landmarks.landmark:
        feats.append(lm.x - wrist_x)
        feats.append(lm.y - wrist_y)
    return np.array(feats, dtype=np.float32)  # (42,)

def next_sample_index(word_dir):
    existing = [f for f in os.listdir(word_dir)
                if f.startswith("sample_") and f.endswith(".npy")
                and "_" not in f.replace("sample_", "", 1).replace(".npy", "")]
    if not existing:
        return 0
    idxs = [int(f.replace("sample_", "").replace(".npy", ""))
            for f in existing if f.replace("sample_", "").replace(".npy", "").isdigit()]
    return max(idxs) + 1 if idxs else 0

def draw_text(img, text, pos, size=0.9, color=(255,255,255), bold=False):
    thickness = 3 if bold else 1
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, (30,30,30), thickness+2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

def between_words_pause(cap, done_word, done_count, next_word, word_idx, total):
    """Show a pause screen between words. Returns True to continue, False to quit."""
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        # dim the camera feed
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (640, 480), (18, 12, 22), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        # ── completed word summary ────────────────────────────────────────
        draw_text(frame, f"[{word_idx}/{total}] Done: {done_word}",
                  (15, 46), size=0.9, color=(80, 220, 130), bold=True)
        draw_text(frame, f"{done_count} new samples saved.",
                  (15, 80), size=0.62, color=(150, 195, 155))

        cv2.line(frame, (15, 106), (625, 106), (90, 60, 100), 1)

        # ── next word ────────────────────────────────────────────────────
        draw_text(frame, "NEXT:", (15, 148), size=0.72, color=(180, 140, 190))
        draw_text(frame, next_word, (15, 222), size=2.2,
                  color=(235, 165, 200), bold=True)

        draw_text(frame, "Take your time to learn the gesture.",
                  (15, 278), size=0.62, color=(200, 175, 205))
        draw_text(frame, "Press SPACE when ready to start recording.",
                  (15, 308), size=0.62, color=(200, 175, 205))

        # ── hint bar ─────────────────────────────────────────────────────
        draw_text(frame, "SPACE = continue    S = skip next    Q = quit",
                  (15, 456), size=0.55, color=(130, 110, 145))

        cv2.imshow("Auralis -- Sample Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            return "continue"
        elif key == ord('s') or key == ord('S'):
            return "skip"
        elif key == ord('q') or key == ord('Q'):
            return "quit"

# ── Main ───────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n" + "="*56)
print("  Auralis -- Gesture Sample Collector")
print("="*56)
print(f"  Words   : {WORDS}")
print(f"  Target  : {SAMPLES_PER_WORD} samples / word")
print(f"  Frames  : {SEQ_LEN} frames / sample")
print("  Controls: SPACE=record  S=skip  Q=quit")
print("="*56 + "\n")

skip_next = False

for word_idx, word in enumerate(WORDS):
    # handle skip requested from pause screen
    if skip_next:
        print(f"  Skipping {word} (requested from pause screen)")
        skip_next = False
        continue

    word_dir = os.path.join(SEQUENCE_DATASET, word)
    os.makedirs(word_dir, exist_ok=True)

    start_idx = next_sample_index(word_dir)
    samples_recorded = 0
    target = SAMPLES_PER_WORD

    print(f"\n[{word_idx+1}/{len(WORDS)}]  Word: {word}  (existing: {start_idx})")

    sample_num = start_idx
    while samples_recorded < target:
        # ── IDLE: waiting for SPACE ────────────────────────────────────────
        waiting = True
        while waiting:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = hands.process(rgb)

            if res.multi_hand_landmarks:
                for hl in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(180,100,120), thickness=2, circle_radius=3),
                        mp_draw.DrawingSpec(color=(220,160,180), thickness=2))

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (640, 100), (30, 20, 35), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            draw_text(frame, f"WORD: {word}", (15, 42), size=1.3, color=(230,160,190), bold=True)
            draw_text(frame, f"Sample {samples_recorded+1}/{target}  |  [{word_idx+1}/{len(WORDS)}]",
                      (15, 78), size=0.6, color=(180,180,180))
            draw_text(frame, "SPACE = record    S = skip word    Q = quit",
                      (15, 455), size=0.55, color=(150,150,150))

            hand_status = "Hand detected" if res.multi_hand_landmarks else "No hand..."
            status_color = (80,220,120) if res.multi_hand_landmarks else (80,80,220)
            draw_text(frame, hand_status, (450, 455), size=0.55, color=status_color)

            cv2.imshow("Auralis -- Sample Collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                waiting = False
            elif key == ord('s') or key == ord('S'):
                print(f"  Skipped {word}")
                samples_recorded = target
                waiting = False
            elif key == ord('q') or key == ord('Q'):
                print("\n  Quitting. Run `python train_lstm_model.py` to retrain.\n")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)

        if samples_recorded >= target:
            break

        # ── COUNTDOWN ────────────────────────────────────────────────────────
        start_time = time.time()
        while time.time() - start_time < COUNTDOWN_SEC:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (640, 100), (20, 20, 50), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            remaining = COUNTDOWN_SEC - (time.time() - start_time)
            draw_text(frame, f"WORD: {word}", (15, 42), size=1.3, color=(230,160,190), bold=True)
            draw_text(frame, f"Get ready...  {remaining:.1f}s", (15, 78),
                      size=0.75, color=(255, 200, 80))
            cv2.imshow("Auralis -- Sample Collector", frame)
            cv2.waitKey(1)

        # ── RECORDING ────────────────────────────────────────────────────────
        sequence    = []
        frames_done = 0
        while frames_done < SEQ_LEN:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = hands.process(rgb)

            feats = None
            if res.multi_hand_landmarks:
                feats = extract_landmarks(res.multi_hand_landmarks[0])
                mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 220, 120), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(0, 160, 80), thickness=2))

            sequence.append(feats if feats is not None else np.zeros(42, dtype=np.float32))
            frames_done += 1

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (640, 100), (20, 40, 20), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            draw_text(frame, f"RECORDING: {word}", (15, 42), size=1.3,
                      color=(80, 255, 140), bold=True)
            prog = frames_done / SEQ_LEN
            cv2.rectangle(frame, (15, 70), (625, 85), (60,60,60), -1)
            cv2.rectangle(frame, (15, 70), (15 + int(610 * prog), 85), (80, 220, 120), -1)
            draw_text(frame, f"{frames_done}/{SEQ_LEN}", (260, 82), size=0.5, color=(255,255,255))
            cv2.imshow("Auralis -- Sample Collector", frame)
            cv2.waitKey(1)

        # ── SAVE ─────────────────────────────────────────────────────────────
        arr = np.array(sequence, dtype=np.float32)
        save_path = os.path.join(word_dir, f"sample_{sample_num}.npy")
        np.save(save_path, arr)
        samples_recorded += 1
        sample_num += 1
        print(f"  Saved sample {samples_recorded}/{target} -> {save_path}")

        # brief saved flash
        for _ in range(15):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            draw_text(frame, "SAVED!", (240, 250), size=2.0, color=(80, 255, 140), bold=True)
            cv2.imshow("Auralis -- Sample Collector", frame)
            cv2.waitKey(1)

    print(f"  Finished {word}  ({samples_recorded} new samples)")

    # ── BETWEEN-WORDS PAUSE ───────────────────────────────────────────────────
    next_word = WORDS[word_idx + 1] if word_idx + 1 < len(WORDS) else None
    if next_word is not None:
        action = between_words_pause(cap, word, samples_recorded,
                                     next_word, word_idx + 1, len(WORDS))
        if action == "quit":
            print("\n  Quitting. Run `python train_lstm_model.py` to retrain.\n")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        elif action == "skip":
            skip_next = True

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*56)
print("  All done! Retrain the model with:")
print("  python train_lstm_model.py")
print("="*56 + "\n")
