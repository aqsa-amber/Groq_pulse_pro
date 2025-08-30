import os
import time
import json
import queue
import threading
import numpy as np
import av
import streamlit as st

# -------------------------------------------------
# Page Config MUST be first
# -------------------------------------------------
st.set_page_config(
    page_title="⚡ GROQ PULSE",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------
# Load secrets from Streamlit's .toml config
# -------------------------------------------------
api_key = st.secrets.get("GROQ_API_KEY", "")
api_url = st.secrets.get("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
tts_enabled_default = st.secrets.get("TTS_ENABLED", False)

st.write("API Key Loaded:", api_key[:5] + "…" if api_key else "⚠️ Not set")

# -------------------------------------------------
# Local utils
# -------------------------------------------------
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.groq_client import GroqClient
from utils.ocr import extract_text_from_file
from utils.tts import speak_text_if_enabled
from utils.db import ConversationStore

# -------------------------------------------------
# WebRTC (browser mic support) - Try Vosk, else Google
# -------------------------------------------------
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

USE_VOSK = False
vosk_model = None

try:
    from vosk import Model, KaldiRecognizer
    model_path = "vosk-model-small-en-us-0.15"
    if os.path.exists(model_path):
        vosk_model = Model(model_path)
        USE_VOSK = True
    else:
        st.warning("⚠️ Vosk model not found, falling back to Google STT.")
except ImportError:
    st.warning("⚠️ Vosk not installed, falling back to Google STT.")

# Fallback STT
import speech_recognition as sr


class AudioProcessor(AudioProcessorBase):
    """
    Collect audio chunks from browser mic.
    If Vosk available → offline STT.
    Else → Google SpeechRecognition online STT.
    """
    def __init__(self):
        self._last_text = None

        if USE_VOSK and vosk_model:
            self.recognizer = KaldiRecognizer(vosk_model, 16000)
            self.q = queue.Queue()
            self.running = True
            threading.Thread(target=self._process_audio, daemon=True).start()
        else:
            self.google_recognizer = sr.Recognizer()

    def recv_audio_frame(self, frame: av.AudioFrame):
        audio = frame.to_ndarray().flatten()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        np.clip(audio, -1.0, 1.0, out=audio)
        pcm16 = (audio * 32767).astype(np.int16).tobytes()

        if USE_VOSK and vosk_model:
            try:
                self.q.put_nowait(pcm16)
            except queue.Full:
                pass
        else:
            # Google fallback (recognizes short chunks)
            with sr.AudioData(pcm16, 16000, 2) as data:
                try:
                    text = self.google_recognizer.recognize_google(data)
                    if text and text != self._last_text:
                        st.session_state["voice_prompt"] = text
                        self._last_text = text
                except sr.UnknownValueError:
                    pass

        return frame

    def _process_audio(self):
        while self.running:
            try:
                data = self.q.get(timeout=1)
            except queue.Empty:
                continue

            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip()
                if text and text != self._last_text:
                    st.session_state["voice_prompt"] = text
                    self._last_text = text
            else:
                partial = json.loads(self.recognizer.PartialResult())
                if partial.get("partial"):
                    st.session_state["voice_prompt"] = partial["partial"]


# -------------------------------------------------
# CSS (themes)
# -------------------------------------------------
if os.path.exists("style.css"):
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -------------------------------------------------
# Session initialization
# -------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "settings" not in st.session_state:
    st.session_state.settings = {
        "temperature": 0.2,
        "max_tokens": 512,
        "top_p": 1.0,
        "model": "llama3-8b-8192"
    }
if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------------------------
# Initialize Groq client and database
# -------------------------------------------------
groq = GroqClient(api_key=api_key, api_url=api_url)
store = ConversationStore(db_path=os.path.join(os.getcwd(), "groq_pulse_db.sqlite"))

# -------------------------------------------------
# Sidebar – Settings
# -------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.session_state.theme = st.radio("Theme", ["light", "dark"])
    st.subheader("Model Parameters")
    st.session_state.settings["model"] = st.selectbox("Model", ["llama3-8b-8192", "llama3-70b-8192"])
    st.session_state.settings["temperature"] = st.slider("Temperature", 0.0, 1.0, float(st.session_state.settings["temperature"]))
    st.session_state.settings["max_tokens"] = st.slider("Max tokens", 64, 4096, int(st.session_state.settings["max_tokens"]))
    st.session_state.settings["top_p"] = st.slider("Top-p", 0.1, 1.0, float(st.session_state.settings["top_p"]))
    st.subheader("Voice Output")
    st.session_state["tts_enabled"] = st.toggle("Speak responses (TTS)", value=tts_enabled_default)

# -------------------------------------------------
# Apply theme wrapper
# -------------------------------------------------
theme_class = "theme-light" if st.session_state.theme == "light" else "theme-dark"
st.markdown(f"<div class='theme {theme_class}'>", unsafe_allow_html=True)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("⚡ GROQ PULSE")
st.caption("Streaming intelligence at the speed of Groq.")

if not api_key:
    st.warning("⚠️ GROQ_API_KEY not set. Running in local fallback mode (no external API).")

# -------------------------------------------------
# Chat History Render
# -------------------------------------------------
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(f"<div class='msg {msg['role']}'>{icon} {msg['content']}</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Input Area
# -------------------------------------------------
with st.form("chat_input", clear_on_submit=True):
    text_in = st.text_area("Send a prompt", key="input_area")
    uploaded = st.file_uploader("Upload doc", type=["txt","pdf","docx","png","jpg","jpeg","csv","json"])
    submitted = st.form_submit_button("Send")

# -------------------------------------------------
# Voice Input (WebRTC)
# -------------------------------------------------
st.subheader("🎙️ Speak your prompt")
webrtc_streamer(
    key="speech",
    mode=WebRtcMode.RECVONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False}
)

if "voice_prompt" in st.session_state:
    st.info(f"🗣️ You said: {st.session_state['voice_prompt']}")
    text_in = st.session_state["voice_prompt"]

# -------------------------------------------------
# Helper
# -------------------------------------------------
def ensure_conversation():
    if st.session_state.current_conversation is None:
        st.session_state.current_conversation = store.create_conversation(name="New Conversation")
        st.session_state.messages = []

# -------------------------------------------------
# Process Input
# -------------------------------------------------
if submitted and text_in.strip():
    ensure_conversation()
    st.session_state.messages.append({"role": "user", "content": text_in})
    response_placeholder = st.empty()
    assistant_content = ""
    try:
        stream_gen = groq.stream_response(text_in, params=st.session_state.settings)
        for chunk in stream_gen:
            assistant_content += getattr(chunk, "text", str(chunk))
            response_placeholder.markdown(f"<div class='msg assistant'>🤖 {assistant_content}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error generating response: {e}")
        assistant_content = "[Error: Could not generate response]"
    st.session_state.messages.append({"role": "assistant", "content": assistant_content})
    store.save_conversation(st.session_state.current_conversation)
    if st.session_state.get("tts_enabled"):
        speak_text_if_enabled(assistant_content)
    st.rerun()

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("<div class='footer'>GROQ PULSE • Powered by Groq</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)











