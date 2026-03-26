"""
Converts sign_language_model.h5 (saved with Keras 3/TF 2.21) to be compatible
with Keras 2.15 by rebuilding the architecture from scratch and copying weights.
"""
import pickle
import numpy as np
import h5py

print("Loading label encoder to get num_classes...")
with open("model/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
num_classes = len(le.classes_)
print(f"  num_classes = {num_classes}")

# Import Keras 2.15 components
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

print("Rebuilding model architecture (Keras 2.15)...")
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
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("  Architecture built.")

print("Loading weights from original H5 file...")
model.load_weights("model/sign_language_model.h5")
print("  Weights loaded successfully.")

print("Saving compatible model to model/sign_language_model.h5 ...")
model.save("model/sign_language_model.h5")
print("  Saved.")

# Verify reload
from keras.models import load_model
m2 = load_model("model/sign_language_model.h5")
print(f"\n✅ Model reloaded successfully! Input: {m2.input_shape}  Output: {m2.output_shape}")
