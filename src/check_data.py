import numpy as np, sys
from pathlib import Path

DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "basic"
for d in sorted(Path(f"data/{DOMAIN}").iterdir()):
    files = list(d.glob("*.npy"))
    missing = []
    for f in files:
        a = np.load(f)
        hands = a[:, 132:]                                  # both hands' keypoints
        missing.append(np.mean(np.all(hands == 0, axis=1))) # frames with no hand
    print(f"{d.name}: {len(files)} files, hands missing in {np.mean(missing):.0%} of frames")