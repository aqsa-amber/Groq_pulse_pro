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
tts_provider = st.secrets.get("TTS_PROVIDER", "pyttsx3")

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
# WebRTC (browser mic support) - Hybrid STT
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
        st.success("✅ Using Vosk offline speech recognition")
    else:
        st.warning("⚠️ Vosk model not found, will use Google SpeechRecognition")
except ImportError:
    st.warning("⚠️ Vosk not installed, using Google SpeechRecognition instead")

# Google fallback
import speech_recognition as sr

# -------------------------------------------------
# AudioProcessor classes
# -------------------------------------------------
class VoskAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.q = queue.Queue()
        self.recognizer = KaldiRecognizer(vosk_model, 16000)
        self.running = True
        self._last_text = None
        self.thread = threading.Thread(target=self._process_audio, daemon=True)
        self.thread.start()

    def recv_audio_frame(self, frame: av.AudioFrame):
        audio = frame.to_ndarray().flatten()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        np.clip(audio, -1.0, 1.0, out=audio)
        pcm16 = (audio * 32767).astype(np.int16).tobytes()
        try:
            self.q.put_nowait(pcm16)
        except queue.Full:
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
                    if st.session_state.get("debug_mode"):
                        st.write("✅ Final:", result)
            else:
                partial = json.loads(self.recognizer.PartialResult())
                if partial.get("partial"):
                    st.session_state["voice_prompt"] = partial["partial"]
                    if st.session_state.get("debug_mode"):
                        st.write("⏳ Partial:", partial)

class GoogleAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def recv_audio_frame(self, frame: av.AudioFrame):
        audio = frame.to_ndarray().flatten().astype("int16")
        audio_data = sr.AudioData(audio.tobytes(), frame.sample_rate, 2)
        try:
            text = self.recognizer.recognize_google(audio_data)
            st.session_state["voice_prompt"] = text
        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            st.error("Google SpeechRecognition API error")
        return frame

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
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

# -------------------------------------------------
# Initialize Groq client and database
# -------------------------------------------------
groq = GroqClient(api_key=api_key, api_url=api_url)
store = ConversationStore(db_path=os.path.join(os.getcwd(), "groq_pulse_db.sqlite"))

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.session_state.theme = st.radio("Theme", ["light", "dark"],
        index=0 if st.session_state.theme == "light" else 1)
    st.subheader("Model Parameters")
    st.session_state.settings["model"] = st.selectbox(
        "Model", ["llama3-8b-8192", "llama3-70b-8192"],
        index=0 if st.session_state.settings["model"] == "llama3-8b-8192" else 1)
    st.session_state.settings["temperature"] = st.slider(
        "Temperature", 0.0, 1.0, float(st.session_state.settings["temperature"]))
    st.session_state.settings["max_tokens"] = st.slider(
        "Max tokens", 64, 4096, int(st.session_state.settings["max_tokens"]))
    st.session_state.settings["top_p"] = st.slider(
        "Top-p", 0.1, 1.0, float(st.session_state.settings["top_p"]))
    st.subheader("Voice Output")
    tts_enabled = st.toggle("Speak responses (TTS)", value=tts_enabled_default)
    st.session_state["tts_enabled"] = tts_enabled
    st.subheader("Debug")
    st.session_state.debug_mode = st.toggle("Show raw Vosk JSON", value=False)

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
        role = msg["role"]
        content = msg["content"]
        bubble = "user" if role == "user" else "assistant"
        icon = "👤" if role == "user" else "🤖"
        st.markdown(f"<div class='msg {bubble}'>{icon} {content}</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Input Area
# -------------------------------------------------
with st.form("chat_input", clear_on_submit=True):
    text_in = st.text_area("Send a prompt", key="input_area")
    uploaded = st.file_uploader("Upload document",
        type=["txt","pdf","docx","png","jpg","jpeg","csv","json"])
    col_send, col_voice = st.columns([3,1])
    submitted = col_send.form_submit_button("Send")

# -------------------------------------------------
# Voice Input (WebRTC)
# -------------------------------------------------
st.subheader("🎙️ Speak your prompt")
webrtc_streamer(
    key="speech",
    mode=WebRtcMode.RECVONLY,
    audio_processor_factory=VoskAudioProcessor if USE_VOSK else GoogleAudioProcessor,
    media_stream_constraints={"audio": True, "video": False}
)

if "voice_prompt" in st.session_state:
    st.info(f"🗣️ You said: {st.session_state['voice_prompt']}")
    text_in = st.session_state["voice_prompt"]

# -------------------------------------------------
# Helper to ensure conversation exists
# -------------------------------------------------
def ensure_conversation():
    if st.session_state.current_conversation is None:
        st.session_state.current_conversation = store.create_conversation(name="New Conversation")
        st.session_state.messages = []

# -------------------------------------------------
# Process User Input
# -------------------------------------------------
if submitted and text_in.strip():
    ensure_conversation()
    if uploaded:
        appended = extract_text_from_file(uploaded, filename=uploaded.name)
        if appended and appended.strip():
            snippet = appended[:4000]
            text_in = f"{text_in}\n\n[Attached document content]\n{snippet}"

    st.session_state.messages.append({"role": "user", "content": text_in})
    st.markdown(f"<div class='msg user'>👤 {text_in}</div>", unsafe_allow_html=True)

    response_placeholder = st.empty()
    assistant_content = ""
    start_time = time.time()
    try:
        stream_gen = groq.stream_response(text_in, params=st.session_state.settings)
        for chunk in stream_gen:
            assistant_content += getattr(chunk, "text", str(chunk))
            response_placeholder.markdown(
                f"<div class='msg assistant'>🤖 {assistant_content}<span class='cursor'>|</span></div>",
                unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error generating response: {e}")
        assistant_content = "[Error: Could not generate response]"
    finally:
        elapsed = time.time() - start_time

    response_placeholder.markdown(
        f"<div class='msg assistant'>🤖 {assistant_content}</div>", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": assistant_content})
    st.session_state.current_conversation["messages"] = st.session_state.messages
    store.save_conversation(st.session_state.current_conversation)
    store.log_interaction(text_in, assistant_content, elapsed,
        conversation_id=st.session_state.current_conversation["id"])

    if st.session_state.get("tts_enabled"):
        speak_text_if_enabled(assistant_content)

    st.rerun()

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("<div class='footer'>GROQ PULSE • Powered by Groq</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)











