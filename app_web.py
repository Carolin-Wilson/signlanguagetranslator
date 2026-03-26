"""
Flask Web Backend for Sign Language Translator
Run: python app_web.py
Open: http://localhost:5000
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
import time
import threading
import collections
import tensorflow as tf
from flask import Flask, Response, jsonify, render_template

# ── Optional spell-check ──────────────────────────────────────────────────────
try:
    from spellchecker import SpellChecker
    spell = SpellChecker()
    SPELL_CHECK_AVAILABLE = True
except ImportError:
    SPELL_CHECK_AVAILABLE = False

# ── Load letter TFLite model (fast, thread-safe) ─────────────────────────────
interpreter = tf.lite.Interpreter(model_path="model/sign_language_model.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("model/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

WORD_LABELS = {cls for cls in le.classes_ if len(cls) > 1}  # empty: model is letter-only

# ── Load word TFLite model (optional — only if trained) ──────────────────────
WORD_MODEL_AVAILABLE = False
word_interpreter   = None
word_input_details = None
word_output_details= None
word_le            = None

if os.path.exists("model/word_model.tflite") and os.path.exists("model/word_label_encoder.pkl"):
    try:
        word_interpreter = tf.lite.Interpreter(model_path="model/word_model.tflite")
        word_interpreter.allocate_tensors()
        word_input_details  = word_interpreter.get_input_details()
        word_output_details = word_interpreter.get_output_details()
        with open("model/word_label_encoder.pkl", "rb") as f:
            word_le = pickle.load(f)
        WORD_MODEL_AVAILABLE = True
        print(f"  ✅  Word model loaded — {len(word_le.classes_)} classes: {list(word_le.classes_)}")
    except Exception as e:
        print(f"  ⚠️  Word model failed to load: {e}")
else:
    print("  ℹ️  Word model not found — running letter-only mode.")
    print("      Train it with: python train_lstm_model.py")

def tflite_predict(landmarks_array):
    """Run single-frame letter inference. Thread-safe (single interpreter)."""
    inp = landmarks_array.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])  # (1, num_letter_classes)

def word_tflite_predict(sequence_array):
    """Run 30-frame word inference. Returns (label, confidence) or (None, 0)."""
    if not WORD_MODEL_AVAILABLE:
        return None, 0.0
    inp = sequence_array.astype(np.float32)          # (1, 30, 42)
    word_interpreter.set_tensor(word_input_details[0]['index'], inp)
    word_interpreter.invoke()
    output = word_interpreter.get_tensor(word_output_details[0]['index'])  # (1, num_words)
    idx  = int(np.argmax(output))
    conf = float(output[0][idx])
    label = word_le.inverse_transform([idx])[0]
    return label, conf

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_draw = mp.solutions.drawing_utils

# ── Shared state ──────────────────────────────────────────────────────────────
lock = threading.Lock()
state = {
    "letter":     "-",
    "confidence": 0.0,
    "is_word":    False,
    "sentence":   "",
    "suggestion": "",
    "hold_pct":   0.0,
    "mode":       "letter",   # "letter" | "word"
}

# Inference mutable state (only touched by inference_loop thread)
current_letter     = ""
letter_start_time  = 0.0
letter_confirmed   = False
no_hand_frames     = 0
pred_buffer        = collections.deque(maxlen=15)
autocorrect_suggestion = ""
autocorrect_timer  = 0.0

# Word recognition state
landmark_buffer    = collections.deque(maxlen=30)  # 30-frame ring buffer

CONFIDENCE_THRESHOLD       = 0.72
WORD_CONFIDENCE_THRESHOLD  = 0.70   # minimum confidence to prefer word over letter
HOLD_TIME_LETTER           = 0.7
HOLD_TIME_WORD             = 1.0
NO_HAND_SPACE_FRAMES       = 30

# ── Camera ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ── Helpers ───────────────────────────────────────────────────────────────────
def try_autocorrect(sentence_str):
    if not SPELL_CHECK_AVAILABLE:
        return ""
    words = sentence_str.strip().split()
    if not words:
        return ""
    last = words[-1].lower()
    if len(last) < 2:
        return ""
    corrected = spell.correction(last)
    return corrected if corrected and corrected != last else ""

# ── Inference loop (background thread) ───────────────────────────────────────
def inference_loop():
    global current_letter, letter_start_time, letter_confirmed
    global no_hand_frames, autocorrect_suggestion, autocorrect_timer

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        predicted_label = "-"
        confidence      = 0.0
        is_word         = False
        hold_pct        = 0.0
        cur_time        = time.time()

        if results.multi_hand_landmarks:
            no_hand_frames = 0

            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(0, 100, 200), thickness=2),
                )

                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x - wrist_x)
                    landmarks.append(lm.y - wrist_y)

                # === Mode-gated inference ===
                with lock:
                    cur_mode = state["mode"]

                if cur_mode == "word" and WORD_MODEL_AVAILABLE:
                    # ── Word-only mode ──────────────────────────────────────
                    landmark_buffer.append(np.array(landmarks, dtype=np.float32))
                    raw_label, raw_conf = "-", 0.0
                    if len(landmark_buffer) == 30:
                        seq = np.array(landmark_buffer, dtype=np.float32).reshape(1, 30, 42)
                        wl, wc = word_tflite_predict(seq)
                        if wl is not None:
                            raw_label, raw_conf = wl, wc
                else:
                    # ── Letter-only mode (default) ──────────────────────────
                    landmarks_flat = np.array(landmarks, dtype=np.float32).reshape(1, -1)
                    prediction = tflite_predict(landmarks_flat)
                    raw_conf  = float(np.max(prediction))
                    raw_label = le.inverse_transform([np.argmax(prediction)])[0]

                # Rolling majority-vote buffer
                pred_buffer.append(raw_label if raw_conf > CONFIDENCE_THRESHOLD else None)
                valid_preds = [p for p in pred_buffer if p is not None]

                buf_len = len(pred_buffer)
                if valid_preds and len(valid_preds) >= buf_len // 2:
                    most_common, votes = collections.Counter(valid_preds).most_common(1)[0]
                    if votes >= buf_len // 3:
                        predicted_label = most_common
                        confidence      = raw_conf
                        is_word         = (cur_mode == "word")

            # Hold-timer: commit letter/word after holding the sign
            hold_required = HOLD_TIME_WORD if is_word else HOLD_TIME_LETTER

            if predicted_label != "-":
                if predicted_label != current_letter:
                    current_letter    = predicted_label
                    letter_start_time = cur_time
                    letter_confirmed  = False
                elif not letter_confirmed:
                    elapsed  = cur_time - letter_start_time
                    hold_pct = min(elapsed / hold_required, 1.0)
                    if elapsed > hold_required:
                        with lock:
                            s = state["sentence"]
                            if is_word:
                                if s and s[-1] != " ":
                                    s += " "
                                s += predicted_label + " "
                            else:
                                s += predicted_label
                            state["sentence"] = s
                        letter_confirmed = True
                        pred_buffer.clear()

        else:
            no_hand_frames += 1
            current_letter   = ""
            letter_confirmed = False
            landmark_buffer.clear()  # reset word buffer when hand disappears

            if no_hand_frames == NO_HAND_SPACE_FRAMES:
                with lock:
                    s = state["sentence"]
                    if s and s[-1] != " ":
                        s += " "
                        state["sentence"] = s
                    suggestion = try_autocorrect(s.rstrip())
                    if suggestion:
                        autocorrect_suggestion = suggestion
                        autocorrect_timer      = cur_time
                        state["suggestion"]    = suggestion

        # Clear old autocorrect
        if autocorrect_suggestion and (cur_time - autocorrect_timer) > 3.0:
            autocorrect_suggestion = ""
            with lock:
                state["suggestion"] = ""

        # ── Overlay on frame ─────────────────────────────────────────────────
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (18, 18, 28), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        label_color = (80, 230, 120) if not is_word else (80, 170, 255)
        mode_text   = "WORD" if is_word else "LETTER"
        cv2.putText(frame, mode_text,        (15, 22),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (160, 160, 160), 1)
        cv2.putText(frame, predicted_label,  (15, 68),  cv2.FONT_HERSHEY_SIMPLEX, 1.8,  label_color, 3)
        cv2.putText(frame, f"{confidence:.0%}", (180, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)

        if current_letter and not letter_confirmed and hold_pct > 0:
            cv2.rectangle(frame, (15, 73), (15 + int(100 * hold_pct), 80), (0, 255, 160), -1)

        with lock:
            state["letter"]     = predicted_label
            state["confidence"] = round(confidence, 3)
            state["is_word"]    = is_word
            state["hold_pct"]   = round(hold_pct, 3)
            # mode is already in state, no need to update here

        # JPEG encode for web streaming
        ret2, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret2:
            with lock:
                app._frame = jpeg.tobytes()

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app._frame = None

def generate_frames():
    while True:
        with lock:
            frame = app._frame
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
def get_state():
    with lock:
        return jsonify(state)

@app.route('/clear', methods=['POST'])
def clear_sentence():
    with lock:
        state["sentence"]   = ""
        state["suggestion"] = ""
    return jsonify({"ok": True})

@app.route('/backspace', methods=['POST'])
def backspace():
    with lock:
        state["sentence"] = state["sentence"][:-1]
    return jsonify({"ok": True})

@app.route('/accept_suggestion', methods=['POST'])
def accept_suggestion():
    with lock:
        sugg = state.get("suggestion", "")
        s    = state["sentence"]
        if sugg and s:
            words = s.rstrip().split()
            if words:
                words[-1]         = sugg
                state["sentence"] = " ".join(words) + " "
        state["suggestion"] = ""
    return jsonify({"ok": True})

@app.route('/set_mode', methods=['POST'])
def set_mode():
    """Switch between 'letter' and 'word' recognition modes."""
    from flask import request
    data = request.get_json(force=True)
    mode = data.get("mode", "letter")
    if mode not in ("letter", "word"):
        return jsonify({"ok": False, "error": "mode must be 'letter' or 'word'"}), 400
    with lock:
        state["mode"]       = mode
        state["sentence"]   = ""   # clear sentence on mode switch
        state["suggestion"] = ""
    # also reset inference buffers (thread-safe via globals)
    pred_buffer.clear()
    landmark_buffer.clear()
    print(f"  Mode switched -> {mode}")
    return jsonify({"ok": True, "mode": mode})

if __name__ == '__main__':
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()
    print("\n  ✅  Sign Language Web App running at http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
