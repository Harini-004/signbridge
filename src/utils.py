import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

def detect(frame_rgb, model):
    frame_rgb.flags.writeable = False
    results = model.process(frame_rgb)
    frame_rgb.flags.writeable = True
    return results

def extract_keypoints(results):
    pose = (np.array([[p.x, p.y, p.z, p.visibility] for p in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(33 * 4))
    lh = (np.array([[p.x, p.y, p.z] for p in results.left_hand_landmarks.landmark]).flatten()
          if results.left_hand_landmarks else np.zeros(21 * 3))
    rh = (np.array([[p.x, p.y, p.z] for p in results.right_hand_landmarks.landmark]).flatten()
          if results.right_hand_landmarks else np.zeros(21 * 3))
    return np.concatenate([pose, lh, rh])   # 258 values per frame