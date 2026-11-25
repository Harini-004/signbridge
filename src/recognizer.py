import json
import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
from utils import mp_holistic, mp_draw, detect, extract_keypoints


class SignRecognizer:
    def __init__(self, domain="basic", frames=30, threshold=0.8):
        self.model = load_model(f"models/{domain}.keras")
        self.labels = json.load(open(f"models/{domain}_labels.json"))
        self.frames, self.threshold = frames, threshold
        self.buffer = deque(maxlen=frames)
        self.recent = deque(maxlen=8)
        self.sentence = []
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process(self, frame):
        results = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self.holistic)
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        if results.left_hand_landmarks or results.right_hand_landmarks:
            self.buffer.append(extract_keypoints(results))
        else:
            self.buffer.clear()
            self.recent.clear()

        if len(self.buffer) == self.frames:
            probs = self.model.predict(np.expand_dims(np.array(self.buffer), 0), verbose=0)[0]
            idx = int(np.argmax(probs))
            self.recent.append(idx if probs[idx] > self.threshold else -1)
            cv2.putText(frame, f"{self.labels[idx]} {probs[idx]:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (55, 175, 212), 2)
            if len(self.recent) == self.recent.maxlen and len(set(self.recent)) == 1 and self.recent[0] != -1:
                word = self.labels[self.recent[0]]
                if not self.sentence or self.sentence[-1] != word:
                    self.sentence.append(word)
        return frame