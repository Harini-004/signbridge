import json, sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "basic"

# 1. Load data (only signs that actually have recordings)
vocab = json.load(open(f"vocab/{DOMAIN}.json"))["signs"]
signs = [s for s in vocab if list(Path(f"data/{DOMAIN}/{s}").glob("*.npy"))]
print("Training on:", signs)

X, y = [], []
for label, sign in enumerate(signs):
    for f in sorted(Path(f"data/{DOMAIN}/{sign}").glob("*.npy")):
        X.append(np.load(f))
        y.append(label)

X = np.array(X)                          # (samples, 30, 258)
y = to_categorical(y).astype(int)        # one-hot labels
print("X shape:", X.shape, "y shape:", y.shape)

# 2. Split into train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Build the model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 258)),
    LSTM(64),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(len(signs), activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 4. Train
early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=200, batch_size=16, callbacks=[early_stop])

# 5. Evaluate
pred = np.argmax(model.predict(X_test), axis=1)
true = np.argmax(y_test, axis=1)
print("\nConfusion matrix:\n", confusion_matrix(true, pred))
print("\n", classification_report(true, pred, target_names=signs))

# 6. Save model + label order
Path("models").mkdir(exist_ok=True)
model.save(f"models/{DOMAIN}.keras")
json.dump(signs, open(f"models/{DOMAIN}_labels.json", "w"))
print(f"Saved models/{DOMAIN}.keras")