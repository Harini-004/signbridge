import re
from pathlib import Path
from moviepy.editor import VideoFileClip, concatenate_videoclips

ASSETS = Path("assets/clips")
LIBRARY = {p.stem.lower(): p for p in ASSETS.glob("*.mp4")}


def clean(text):
    text = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return text.split()


def plan(text):
    """Turn text into an ordered list of (label, clip_path)."""
    words = clean(text)
    steps, i = [], 0
    while i < len(words):
        pair = " ".join(words[i:i + 2])
        if len(words[i:i + 2]) == 2 and pair in LIBRARY:      # two-word sign
            steps.append((pair, LIBRARY[pair]))
            i += 2
            continue
        word = words[i]
        if word in LIBRARY:                                    # known word
            steps.append((word, LIBRARY[word]))
        else:                                                  # fingerspell it
            for ch in word:
                if ch in LIBRARY:
                    steps.append((ch, LIBRARY[ch]))
        i += 1
    return steps


def build_video(text, out="output/result.mp4"):
    steps = plan(text)
    if not steps:
        return None, []
    Path(out).parent.mkdir(exist_ok=True)
    clips = [VideoFileClip(str(path)) for _, path in steps]
    size = clips[0].size
    clips = [c if c.size == size else c.resize(newsize=size) for c in clips]
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(out, codec="libx264", audio=False, logger=None)
    for c in clips:
        c.close()
    return out, [label for label, _ in steps]


if __name__ == "__main__":
    print(f"{len(LIBRARY)} clips loaded")
    path, labels = build_video("hello nice to meet you 25")
    print("Played:", labels, "->", path)