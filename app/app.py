import sys, time
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

sys.path.append("src")
from text_to_sign import build_video
from recognizer import SignRecognizer
import base64
import json
import streamlit.components.v1 as components
from streamlit_mic_recorder import speech_to_text
import os
DEPLOYED = os.path.exists("/mount/src")   # True only on Streamlit Community Cloud

def speak(text):
    # speaks through the browser's built-in voice; the comment makes each call unique
    components.html(f"""
    <script>
      const u = new SpeechSynthesisUtterance({json.dumps(text)});
      u.lang = "en-IN"; u.rate = 0.95;
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(u);
    </script><!-- {time.time()} -->""", height=0)

def bg_image(path):
    data = base64.b64encode(open(path, "rb").read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{data}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>""", unsafe_allow_html=True)

st.set_page_config(page_title="SignBridge", page_icon="🤟", layout="wide")
bg_image("assets/bg.png")
st.markdown("""
<style>
:root { --maroon:#7A1F2B; --terracotta:#B5412A; --gold:#D9A441;
        --cream:#FFF8EE; --sand:#F6E7CF; --ink:#3B1F1A; }

.block-container { padding-top: 1.2rem; padding-bottom: 0.5rem; max-width: 1250px; }
#MainMenu, footer { visibility: hidden; }

h1 { color: var(--maroon); font-weight: 800; margin-bottom: 0; }
h3 { color: var(--maroon); }
.tagline { color: var(--terracotta); margin-bottom: .6rem; font-size: 1.8rem; }

.stTabs [data-baseweb="tab"] { font-size: 1.1rem; font-weight: 600; color: var(--ink); }
.stTabs [aria-selected="true"] { color: var(--maroon); }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--gold); }

.stButton > button { background: var(--maroon); border: 0; border-radius: 10px;
                     padding: .6rem 1.4rem; font-weight: 600; }
.stButton > button p { color: #fff; font-size: 1.05rem; }
.stButton > button:hover { background: var(--terracotta); }

.card { background: var(--sand); border-left: 6px solid var(--gold);
        border-radius: 12px; padding: 1rem 1.2rem; color: var(--ink); }
.result { font-size: 2rem; font-weight: 700; color: var(--maroon); min-height: 3.5rem; }
.hint { font-size: .95rem; }

video { max-height: 58vh; object-fit: contain; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

st.markdown("# SignBridge")
st.markdown('<div class="tagline">Two-way sign language translation</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Sign → Text", "Text → Sign"])


class Processor(VideoProcessorBase):
    def __init__(self):
        self.rec = SignRecognizer("basic")

    def recv(self, frame):
        img = self.rec.process(frame.to_ndarray(format="bgr24"))
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------- Text → Sign (written first so the live loop below can't block it) ----------
with tab2:
    st.session_state.setdefault("msg", "hello nice to meet you")
    result = st.session_state.get("result")
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.subheader("Speak or type")
        spoken = speech_to_text(language="en-IN", start_prompt="🎤 Speak", stop_prompt="⏹ Stop",
                                just_once=True, use_container_width=True, key="stt")
        if spoken:
            st.session_state["msg"] = spoken
            with st.spinner("Building video..."):
                st.session_state["result"] = build_video(spoken)
            result = st.session_state["result"]

        text = st.text_area("Message", key="msg", height=110, label_visibility="collapsed")
        if st.button("Show signs", key="go"):
            if text.strip():
                with st.spinner("Building video..."):
                    st.session_state["result"] = build_video(text)
                result = st.session_state["result"]
            else:
                st.warning("Type something first.")
        if result and result[1]:
            st.markdown('<div class="card hint"><b>Signed:</b> ' + " → ".join(result[1]) + "</div>",
                        unsafe_allow_html=True)

    with right:
        if result and result[0]:
            with open(result[0], "rb") as f:
                st.video(f.read())
        else:
            st.markdown('<div class="card hint">Your sign video will appear here.</div>',
                        unsafe_allow_html=True)

# ---------- Sign → Text ----------
with tab1:
    if DEPLOYED:
        st.info("Live camera recognition needs a relay server that free hosting doesn't provide. "
                "Here it is running; to try it yourself, run it locally (steps are in the GitHub README).")
        st.video("docs/app_demo.mp4")
        st.stop()
    left, right = st.columns([1.2, 1], gap="large")

    with left:
        ctx = webrtc_streamer(
            key="sign", video_processor_factory=Processor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )

    with right:
        st.subheader("Recognized text")
        box = st.empty()
        box.markdown('<div class="card result">…</div>', unsafe_allow_html=True)
        auto = st.checkbox("Read words aloud automatically", value=True)
        c1, c2 = st.columns(2)
        clear = c1.button("Clear text", key="clear")
        read = c2.button("🔊 Read aloud", key="read")
        speak_slot = st.empty()
        st.markdown('<div class="card hint"><b>Tips</b><br>'
                    '• Face a light source, not a window behind you<br>'
                    '• Keep both hands inside the frame<br>'
                    '• Hold each sign steadily for about a second</div>',
                    unsafe_allow_html=True)

    if ctx.video_processor:
        rec = ctx.video_processor.rec
        if clear:
            rec.sentence.clear()
        if read and rec.sentence:
            with speak_slot:
                speak(" ".join(rec.sentence))

    spoken_n = 0
    while ctx.state.playing:
        if ctx.video_processor:
            sent = ctx.video_processor.rec.sentence
            box.markdown(f'<div class="card result">{" ".join(sent) or "…"}</div>',
                         unsafe_allow_html=True)
            if auto and len(sent) > spoken_n:
                with speak_slot:
                    speak(sent[-1])
            spoken_n = len(sent)
        time.sleep(0.3)