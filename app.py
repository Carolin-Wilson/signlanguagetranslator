import cv2
import numpy as np
import mediapipe as mp
import pickle
import time
import collections
from keras.models import load_model

# ── Try to import spell checker (optional dependency) ─────────────────────────
try:
    from spellchecker import SpellChecker
    spell = SpellChecker()
    SPELL_CHECK_AVAILABLE = True
except ImportError:
    SPELL_CHECK_AVAILABLE = False

# ── Known word-level gesture labels (multi-character labels = words) ───────────
# Any label longer than 1 character is treated as a whole word.
# Examples: HELLO, YES, NO, PLEASE, SORRY, HELP, THANKYOU, GOOD, BAD, STOP

# ── Load model and label encoder ──────────────────────────────────────────────
model = load_model("model/sign_language_model.h5")
with open("model/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

WORD_LABELS = {cls for cls in le.classes_ if len(cls) > 1}
LETTER_LABELS = {cls for cls in le.classes_ if len(cls) == 1}

print(f"Model loaded. Classes: {sorted(le.classes_)}")
print(f"Word gestures: {sorted(WORD_LABELS)}")
print(f"Letter gestures: {sorted(LETTER_LABELS)}")

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ── Camera ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)

# ── State variables ───────────────────────────────────────────────────────────
sentence = ""
current_letter = ""
letter_start_time = 0
letter_confirmed = False
no_hand_frames = 0

# Suggestion shown after spell-check autocorrect
autocorrect_suggestion = ""
autocorrect_timer = 0
AUTOCORRECT_DISPLAY_DURATION = 3.0  # seconds to show correction

# Rolling prediction buffer for stabilization (majority vote)
BUFFER_SIZE = 12
pred_buffer = collections.deque(maxlen=BUFFER_SIZE)

# Thresholds
CONFIDENCE_THRESHOLD = 0.82
HOLD_TIME_LETTER = 0.7    # seconds to confirm a letter
HOLD_TIME_WORD = 1.0      # slightly longer hold for whole-word gestures
NO_HAND_SPACE_FRAMES = 35

# FPS tracking
fps_counter = collections.deque(maxlen=30)
last_frame_time = time.time()

# ── Helper: draw rounded rectangle ───────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(img, (cx, cy), radius, color, thickness)

# ── Helper: spell-check last word in sentence ─────────────────────────────────
def try_autocorrect(sentence_str):
    if not SPELL_CHECK_AVAILABLE:
        return None
    words = sentence_str.strip().split()
    if not words:
        return None
    last = words[-1].lower()
    if len(last) < 2:
        return None
    corrected = spell.correction(last)
    if corrected and corrected != last:
        return corrected
    return None

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS
    now = time.time()
    fps_counter.append(1.0 / max(now - last_frame_time, 1e-6))
    last_frame_time = now
    fps = int(np.mean(fps_counter))

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    predicted_label = "-"
    confidence = 0.0
    is_word = False

    if results.multi_hand_landmarks:
        no_hand_frames = 0

        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=3),
                                   mp_draw.DrawingSpec(color=(0, 100, 200), thickness=2))

            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x - wrist_x)
                landmarks.append(lm.y - wrist_y)

            landmarks = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(landmarks, verbose=0)
            raw_confidence = float(np.max(prediction))
            raw_label = le.inverse_transform([np.argmax(prediction)])[0]

            # Add to rolling buffer only if confident
            if raw_confidence > CONFIDENCE_THRESHOLD:
                pred_buffer.append(raw_label)
            else:
                pred_buffer.append(None)

            # Majority vote from buffer
            valid_preds = [p for p in pred_buffer if p is not None]
            if valid_preds and len(valid_preds) >= BUFFER_SIZE // 2:
                from collections import Counter
                most_common, count_votes = Counter(valid_preds).most_common(1)[0]
                if count_votes >= BUFFER_SIZE // 3:
                    predicted_label = most_common
                    confidence = raw_confidence
                    is_word = predicted_label in WORD_LABELS

            current_time = time.time()
            hold_required = HOLD_TIME_WORD if is_word else HOLD_TIME_LETTER

            if predicted_label != "-":
                if predicted_label != current_letter:
                    current_letter = predicted_label
                    letter_start_time = current_time
                    letter_confirmed = False

                elif (current_time - letter_start_time) > hold_required and not letter_confirmed:
                    if is_word:
                        # Insert whole word with surrounding spaces
                        if sentence and sentence[-1] != " ":
                            sentence += " "
                        sentence += predicted_label + " "
                    else:
                        sentence += predicted_label
                    letter_confirmed = True
                    pred_buffer.clear()

    else:
        no_hand_frames += 1
        current_letter = ""
        letter_confirmed = False

        if no_hand_frames == NO_HAND_SPACE_FRAMES:
            if sentence and sentence[-1] != " ":
                sentence += " "
                # Try autocorrect on the last finger-spelled word
                suggestion = try_autocorrect(sentence.rstrip())
                if suggestion:
                    autocorrect_suggestion = suggestion
                    autocorrect_timer = time.time()

    # ── Dismiss old autocorrect suggestion ────────────────────────────────────
    if autocorrect_suggestion and (time.time() - autocorrect_timer) > AUTOCORRECT_DISPLAY_DURATION:
        autocorrect_suggestion = ""

    # ══════════════════════════════════════════════════════════════════════════
    # UI RENDERING
    # ══════════════════════════════════════════════════════════════════════════

    # ── Top panel ─────────────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (18, 18, 28), -1)
    cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)

    # ── Detected label (Letter or Word) ───────────────────────────────────────
    label_color = (80, 230, 120) if not is_word else (80, 170, 255)
    mode_text = "WORD" if is_word else "LETTER"
    mode_color = (80, 170, 255) if is_word else (180, 180, 180)

    cv2.putText(frame, mode_text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1)
    cv2.putText(frame, predicted_label,
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, label_color, 3)

    # ── Confidence bar ────────────────────────────────────────────────────────
    conf_bar_x, conf_bar_y = 160, 55
    conf_bar_w, conf_bar_h = 180, 14
    cv2.rectangle(frame, (conf_bar_x, conf_bar_y),
                  (conf_bar_x + conf_bar_w, conf_bar_y + conf_bar_h), (60, 60, 60), -1)
    fill = int(conf_bar_w * confidence)
    bar_col = (0, 220, 100) if confidence > 0.9 else (0, 180, 255) if confidence > 0.7 else (0, 80, 200)
    cv2.rectangle(frame, (conf_bar_x, conf_bar_y),
                  (conf_bar_x + fill, conf_bar_y + conf_bar_h), bar_col, -1)
    cv2.putText(frame, f"Conf: {confidence:.0%}", (conf_bar_x, conf_bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ── Hold-progress bar (fills as letter/word is being confirmed) ───────────
    if current_letter and current_letter != "-" and not letter_confirmed:
        hold_required = HOLD_TIME_WORD if current_letter in WORD_LABELS else HOLD_TIME_LETTER
        elapsed = min(time.time() - letter_start_time, hold_required)
        hold_frac = elapsed / hold_required

        hold_x, hold_y = 20, 95
        hold_w = 120
        hold_h = 8
        cv2.rectangle(frame, (hold_x, hold_y), (hold_x + hold_w, hold_y + hold_h), (50, 50, 50), -1)
        fill_w = int(hold_w * hold_frac)
        cv2.rectangle(frame, (hold_x, hold_y), (hold_x + fill_w, hold_y + hold_h),
                      (0, 255, 160), -1)

    # ── FPS ───────────────────────────────────────────────────────────────────
    cv2.putText(frame, f"FPS: {fps}", (w - 110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

    # ── Bottom panel ──────────────────────────────────────────────────────────
    panel_h = 135
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - panel_h), (w, h), (18, 18, 28), -1)
    cv2.addWeighted(overlay2, 0.92, frame, 0.08, 0, frame)

    # ── Sentence display (truncate if too wide) ────────────────────────────────
    display_sentence = sentence
    font_scale = 0.85
    font_thickness = 2
    max_chars = 42  # approx chars that fit at this font size
    if len(display_sentence) > max_chars:
        display_sentence = "..." + display_sentence[-(max_chars - 3):]

    cv2.putText(frame, "Sentence:", (20, h - panel_h + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
    cv2.putText(frame, display_sentence,
                (20, h - panel_h + 65),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    # ── Autocorrect suggestion ─────────────────────────────────────────────────
    if autocorrect_suggestion:
        pill_text = f"  Did you mean: {autocorrect_suggestion}?  [A] Accept"
        cv2.rectangle(frame, (18, h - 60), (18 + len(pill_text) * 9, h - 35), (30, 80, 160), -1)
        cv2.putText(frame, pill_text, (24, h - 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1)

    # ── Key bindings legend ───────────────────────────────────────────────────
    legend = "[Q] Quit  [C] Clear  [Bksp] Delete  [A] Accept suggestion"
    cv2.putText(frame, legend, (20, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)

    cv2.imshow("Sign Language Translator  |  Letter + Word Mode", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('c'):
        sentence = ""
        autocorrect_suggestion = ""
        pred_buffer.clear()

    elif key == 8:   # Backspace
        sentence = sentence[:-1]

    elif key == ord('a') and autocorrect_suggestion:
        # Accept autocorrect suggestion — replace last word
        words = sentence.rstrip().split()
        if words:
            words[-1] = autocorrect_suggestion
            sentence = " ".join(words) + " "
        autocorrect_suggestion = ""

cap.release()
cv2.destroyAllWindows()