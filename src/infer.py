import cv2, json, sys
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
from utils import mp_holistic, mp_draw, detect, extract_keypoints

DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "basic"
model = load_model(f"models/{DOMAIN}.keras")
labels = json.load(open(f"models/{DOMAIN}_labels.json"))

FRAMES = 30        # must match collect_data.py
THRESHOLD = 0.9    # minimum confidence to accept a prediction
buffer = deque(maxlen=FRAMES)   # last 30 frames of keypoints
recent = deque(maxlen=8)        # last 8 predictions, for smoothing
sentence = []

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        results = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), holistic)
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        hands_visible = results.left_hand_landmarks or results.right_hand_landmarks
        conf_text = ""

        if hands_visible:
            buffer.append(extract_keypoints(results))
        else:
            buffer.clear()
            recent.clear()

        if len(buffer) == FRAMES:
            probs = model.predict(np.expand_dims(np.array(buffer), axis=0), verbose=0)[0]
            idx = int(np.argmax(probs))
            conf_text = f"{labels[idx]} {probs[idx]:.2f}"
            recent.append(idx if probs[idx] > THRESHOLD else -1)

            if len(recent) == recent.maxlen and len(set(recent)) == 1 and recent[0] != -1:
                word = labels[recent[0]]
                if not sentence or sentence[-1] != word:
                    sentence.append(word)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (30, 30, 30), -1)
        cv2.putText(frame, " ".join(sentence[-5:]), (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, conf_text, (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.imshow("SignBridge", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            sentence.clear()

cap.release()
cv2.destroyAllWindows()