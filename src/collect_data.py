import cv2, json, sys, numpy as np
from pathlib import Path
from utils import mp_holistic, mp_draw, detect, extract_keypoints

DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "basic"
SEQUENCES, FRAMES = 40, 30

signs = json.load(open(f"vocab/{DOMAIN}.json"))["signs"]
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for sign in signs:
        folder = Path(f"data/{DOMAIN}/{sign}")
        folder.mkdir(parents=True, exist_ok=True)
        for seq in range(SEQUENCES):
            if (folder / f"{seq}.npy").exists():      # resume support
                continue
            # get-ready pause
            for _ in range(40):
                ok, frame = cap.read()
                cv2.putText(frame, f"GET READY: {sign}  ({seq+1}/{SEQUENCES})", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.imshow("collect", frame)
                cv2.waitKey(30)
            # record
            keypoints = []
            for _ in range(FRAMES):
                ok, frame = cap.read()
                results = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), holistic)
                keypoints.append(extract_keypoints(results))
                mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                cv2.putText(frame, f"RECORDING {sign}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow("collect", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap.release(); cv2.destroyAllWindows(); sys.exit()
            np.save(folder / f"{seq}.npy", np.array(keypoints))

cap.release()
cv2.destroyAllWindows()