import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

DATASET_PATH = "dataset"

X = []
y = []

print("\n" + "="*50)
print("  Sign Language Model Trainer")
print("="*50)
print("\nReading dataset...")

for file in sorted(os.listdir(DATASET_PATH)):
    if file.endswith(".csv"):
        label = file.split(".")[0]
        file_path = os.path.join(DATASET_PATH, file)
        data = pd.read_csv(file_path, header=None)
        # coerce any non-numeric cell (e.g. header strings like 'q=0.0') to NaN
        data = data.apply(pd.to_numeric, errors='coerce')
        valid_rows = data.dropna()
        for row in valid_rows.values:
            if len(row) == 42:   # 21 landmarks × 2 coords
                X.append(row)
                y.append(label)
        print(f"  [{label}]  {len(valid_rows)} samples")

X = np.array(X, dtype=np.float32)
y = np.array(y)

print(f"\n  Total samples : {len(X)}")
print(f"  Total classes : {len(set(y))}  →  {sorted(set(y))}\n")

# ── Label encoding ────────────────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# ── Train/test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"  Training set  : {len(X_train)} samples")
print(f"  Test set      : {len(X_test)} samples\n")

# ── Model ─────────────────────────────────────────────────────────────────────
num_classes = len(le.classes_)

model = Sequential([
    Dense(512, activation='relu', input_shape=(42,)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True,
                  verbose=1),
    ModelCheckpoint("model/sign_language_model.h5", monitor='val_accuracy',
                    save_best_only=True, verbose=1)
]

# ── Training ──────────────────────────────────────────────────────────────────
print("\nTraining model...\n")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# ── Evaluation ────────────────────────────────────────────────────────────────
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n  ✅  Final Test Accuracy : {accuracy * 100:.2f}%")

# ── Per-class accuracy ────────────────────────────────────────────────────────
print("\n  Per-class accuracy:")
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

for idx, cls in enumerate(le.classes_):
    mask = y_true == idx
    if mask.sum() > 0:
        cls_acc = (y_pred[mask] == idx).mean() * 100
        print(f"    {cls:12s} : {cls_acc:.1f}%  ({mask.sum()} samples)")

# ── Save artefacts ────────────────────────────────────────────────────────────
# Model already saved by ModelCheckpoint
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

np.save("model/labels.npy", le.classes_)

print("\n  Model and label encoder saved to model/")
print("  Run `python app.py` to use the updated model.\n")