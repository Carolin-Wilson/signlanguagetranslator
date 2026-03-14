import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

DATASET_PATH = "dataset"

X = []
y = []

print("Reading dataset...")

for file in os.listdir(DATASET_PATH):
    if file.endswith(".csv"):
        label = file.split(".")[0]
        file_path = os.path.join(DATASET_PATH, file)

        data = pd.read_csv(file_path, header=None)

        for row in data.values:
            X.append(row)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset loaded.")
print("Total samples:", len(X))

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

print("Training model...")

model = Sequential([
    Dense(256, activation='relu', input_shape=(42,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=80,
    validation_data=(X_test, y_test)
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")

# Save model
if not os.path.exists("model"):
    os.makedirs("model")

model.save("model/sign_language_model.h5")

import pickle
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Model and label encoder saved successfully!")

# Save label encoder
np.save("model/labels.npy", le.classes_)

print("Model saved successfully!")