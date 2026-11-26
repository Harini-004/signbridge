# SignBridge

Two-way sign language translation with voice. Sign → text → speech, and speech/text → sign.

## Demo
▶ [Watch the full demo](docs/demo.mp4)

**Live app (Text/Speech → Sign):** https://YOUR-APP-LINK.streamlit.app
Live camera recognition runs locally; see Setup below.

## Why
Built to explore how computer vision and speech tools can make everyday communication easier between deaf and hearing people, starting with a small Sign Language vocabulary.

## Features
- **Sign → Text:** live webcam recognition using MediaPipe landmarks and an LSTM classifier, trained on ISL signs I recorded myself
- **Sign → Speech:** recognized words are read aloud in the browser
- **Speech → Sign:** speak or type a sentence and the matching sign clips play
- **Fingerspelling fallback:** words without a clip are spelled letter by letter
- **Domain packs:** new vocabularies (healthcare, education, public services...) are added through a JSON file in `vocab/`, with no code changes

## How it works
```
Webcam → MediaPipe Holistic (258 keypoints/frame) → 30-frame buffer → LSTM → word → text + speech
Speech/Text → word lookup → avatar clips (A–Z fingerspelling fallback) → video
```

## Results
- Vocabulary: 4 ISL signs (hello, nice, meet, help), 40 recorded sequences each (160 total)
- Held-out test accuracy: 100% on 32 sequences (8 per sign)
- Informal testing: other people tried it live and it recognised their signs well; a formal multi-signer evaluation is planned
- Caveat: the test split comes from the same session and signer as the training data, so 100% likely overestimates real-world accuracy

## Limitations
- Small vocabulary, trained on one signer
- Sensitive to lighting and hands leaving the frame
- Browser speech recognition needs Chrome or Edge and an internet connection

## Setup
Python 3.11 is required (TensorFlow and MediaPipe don't support newer versions yet).

```powershell
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Collect data, train, and run:
```powershell
python src/collect_data.py basic
python src/train.py basic
streamlit run app/app.py
```

For the Text → Sign tab, add sign video clips named after the word (e.g. `hello.mp4`, `A.mp4`, `5.mp4`) to `assets/clips/`. The clips are not included in this repo;

## Project structure
```
vocab/    sign vocabularies (domain packs)
src/      data collection, training, recognizer, text-to-sign
app/      Streamlit app
models/   trained model and label files
```

## Roadmap
- Record more signers and report accuracy on unseen people
- Landmark normalisation and data augmentation
- Larger vocabulary and continuous (sentence-level) signing
- Domain packs, e.g. healthcare, education, and public services
- Public deployment

