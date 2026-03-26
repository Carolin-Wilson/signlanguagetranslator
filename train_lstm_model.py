"""
train_lstm_model.py
Train an LSTM model on 30-frame hand-landmark sequences for word recognition.
Outputs:
  model/word_model.h5
  model/word_model.tflite
  model/word_label_encoder.pkl
  model/word_labels.npy
"""

import os
import numpy as np
import pickle
import collections
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf

SEQUENCE_DATASET = "sequence_dataset"
SEQ_LEN          = 30   # frames per sample
FEAT_DIM         = 42   # 21 landmarks × 2 (x, y)
BATCH_SIZE       = 32
MAX_EPOCHS       = 100

print("\n" + "="*56)
print("  Auralis -- LSTM Word Model Trainer")
print("="*56)

# ── Load data ─────────────────────────────────────────────────
X, y = [], []
class_counts = collections.Counter()

for word in sorted(os.listdir(SEQUENCE_DATASET)):
    word_dir = os.path.join(SEQUENCE_DATASET, word)
    if not os.path.isdir(word_dir):
        continue
    for fname in os.listdir(word_dir):
        if not fname.endswith(".npy"):
            continue
        arr = np.load(os.path.join(word_dir, fname))
        # Ensure correct shape (30, 42)
        if arr.shape == (SEQ_LEN, FEAT_DIM):
            X.append(arr)
            y.append(word)
            class_counts[word] += 1
        elif arr.ndim == 2 and arr.shape[1] == FEAT_DIM:
            # Pad or truncate to SEQ_LEN
            if arr.shape[0] < SEQ_LEN:
                pad = np.zeros((SEQ_LEN - arr.shape[0], FEAT_DIM), dtype=np.float32)
                arr = np.vstack([arr, pad])
            else:
                arr = arr[:SEQ_LEN]
            X.append(arr)
            y.append(word)
            class_counts[word] += 1

X = np.array(X, dtype=np.float32)   # (N, 30, 42)
y = np.array(y)

print(f"\n  Total samples : {len(X)}")
print(f"  Total classes : {len(class_counts)}  ->  {sorted(class_counts)}")
print(f"\n  Samples per class:")
for cls, cnt in sorted(class_counts.items()):
    print(f"    {cls:14s}: {cnt}")

if len(X) == 0:
    raise RuntimeError("No sequence data found in sequence_dataset/. "
                       "Run extract_wlasl_landmarks.py first.")

# ── Label encoding ────────────────────────────────────────────
le = LabelEncoder()
y_enc  = le.fit_transform(y)
y_cat  = to_categorical(y_enc)
num_classes = len(le.classes_)

# ── Train / test split ────────────────────────────────────────
X_train, X_test, y_train, y_test, y_enc_train, y_enc_test = train_test_split(
    X, y_cat, y_enc,
    test_size=0.2, random_state=42, stratify=y_enc
)
print(f"\n  Train : {len(X_train)} samples")
print(f"  Test  : {len(X_test)}  samples\n")

# ── Model ─────────────────────────────────────────────────────
model = Sequential([
    LSTM(128, input_shape=(SEQ_LEN, FEAT_DIM), return_sequences=True),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(num_classes, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── Callbacks ─────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint("model/word_model.h5", monitor='val_accuracy',
                    save_best_only=True, verbose=0),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                      verbose=1, min_lr=1e-5),
]

# ── Training ──────────────────────────────────────────────────
print("\nTraining LSTM model...\n")
history = model.fit(
    X_train, y_train,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1,
)

# ── Evaluation ────────────────────────────────────────────────
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n  [OK] Final Test Accuracy : {accuracy * 100:.2f}%")

# Per-class accuracy
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true = y_enc_test
print("\n  Per-class accuracy:")
for idx, cls in enumerate(le.classes_):
    mask = y_true == idx
    if mask.sum() > 0:
        cls_acc = (y_pred[mask] == idx).mean() * 100
        print(f"    {cls:14s}: {cls_acc:5.1f}%  ({mask.sum()} test samples)")

# ── Save Keras model & label artifacts ───────────────────────
model.save("model/word_model.h5")
with open("model/word_label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
np.save("model/word_labels.npy", le.classes_)
print("  Saved: model/word_model.h5")
print("  Saved: model/word_label_encoder.pkl")
print("  Saved: model/word_labels.npy")

# ── Convert to TFLite via SavedModel (more reliable on Windows) ──────────────
print("\n  Converting to TFLite...")
import tempfile, shutil
tmp_dir = tempfile.mkdtemp()
try:
    model.export(tmp_dir)   # TF2 SavedModel format
except AttributeError:
    # Older Keras: use tf.saved_model.save
    tf.saved_model.save(model, tmp_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter._experimental_lower_tensor_list_ops = False
tflite_model = converter.convert()
shutil.rmtree(tmp_dir, ignore_errors=True)

with open("model/word_model.tflite", "wb") as f:
    f.write(tflite_model)
print("  Saved: model/word_model.tflite")

# Quick shape check
interp = tf.lite.Interpreter(model_path="model/word_model.tflite")
interp.allocate_tensors()
inp_shape = interp.get_input_details()[0]['shape']
print(f"  TFLite input shape: {inp_shape}  [OK]")
print("\n  Run `python app_web.py` to use the updated model.\n")
