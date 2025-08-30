import os
import time
import json
import queue
import threading
import numpy as np
import av
import streamlit as st
import zipfile
import urllib.request

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

from utils.groq_client import GroqClient, StreamChunk
from utils.ocr import extract_text_from_file
from utils.tts import speak_text_if_enabled
from utils.db import ConversationStore

# -------------------------------------------------
# WebRTC (browser mic support) - now using Vosk
# -------------------------------------------------
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
from vosk import Model, KaldiRecognizer

# -------------------------------------------------
# Auto-download Vosk model if missing
# -------------------------------------------------
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
MODEL_DIR = "vosk-model-small-en-us-0.15"

if "vosk_model" not in st.session_state:
    if not os.path.exists(MODEL_DIR):
        st.warning("⬇️ Downloading Vosk model (first run may take a minute)...")
        zip_path = "model.zip"
        urllib.request.urlretrieve(MODEL_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")
        os.remove(zip_path)
        st.success("✅ Vosk model downloaded!")
    st.session_state.vosk_model = Model(MODEL_DIR)

vosk_model = st.session_state.vosk_model


class AudioProcessor(AudioProcessorBase):
    """
    Collect audio chunks from browser mic,
    resample to 16kHz PCM16, and process with Vosk in background.
    """
    def __init__(self):
        self.q = queue.Queue()
        self.recognizer = KaldiRecognizer(vosk_model, 16000)
        self.running = True
        self._last_text = None

        # Start background thread for recognition
        self.thread = threading.Thread(target=self._process_audio, daemon=True)
        self.thread.start()

    def recv_audio_frame(self, frame: av.AudioFrame):
        # Convert float32 → int16 PCM
        audio = frame.to_ndarray().flatten()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        np.clip(audio, -1.0, 1.0, out=audio)
        pcm16 = (audio * 32767).astype(np.int16).tobytes()

        # Push to queue (non-blocking)
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
# Sidebar – Settings & Conversations
# -------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    st.session_state.theme = st.radio(
        "Theme", ["light", "dark"],
        index=0 if st.session_state.theme == "light" else 1
    )

    st.subheader("Model Parameters")
    st.session_state.settings["model"] = st.selectbox(
        "Model", ["llama3-8b-8192", "llama3-70b-8192"],
        index=0 if st.session_state.settings["model"] == "llama3-8b-8192" else 1
    )
    st.session_state.settings["temperature"] = st.slider(
        "Temperature", 0.0, 1.0, float(st.session_state.settings["temperature"])
    )
    st.session_state.settings["max_tokens"] = st.slider(
        "Max tokens", 64, 4096, int(st.session_state.settings["max_tokens"])
    )
    st.session_state.settings["top_p"] = st.slider(
        "Top-p", 0.1, 1.0, float(st.session_state.settings["top_p"])
    )

    st.subheader("Voice Output")
    tts_enabled = st.toggle("Speak responses (TTS)", value=tts_enabled_default)
    st.session_state["tts_enabled"] = tts_enabled

    st.subheader("Debug")
    st.session_state.debug_mode = st.toggle("Show raw Vosk JSON", value=False)

    st.subheader("Conversations 📜")
    conversations = store.list_conversations()
    if conversations:
        names = [f"{c['id']}: {c['name']}" for c in conversations]
        sel = st.selectbox("Select", options=names)
        sel_id = int(sel.split(":")[0]) if sel else None

        if st.session_state.current_conversation is None or st.session_state.current_conversation.get("id") != sel_id:
            conv = store.get_conversation(sel_id)
            st.session_state.current_conversation = conv
            st.session_state.messages = conv["messages"]

        with st.expander("Manage conversation"):
            new_name = st.text_input(
                "Rename to",
                value=st.session_state.current_conversation.get("name", "Conversation")
            )
            cols = st.columns(3)
            if cols[0].button("💾 Rename"):
                store.rename_conversation(st.session_state.current_conversation["id"], new_name)
                st.session_state.current_conversation["name"] = new_name
                st.success("Renamed.")
                st.rerun()
            if cols[1].button("🗑️ Delete"):
                store.delete_conversation(st.session_state.current_conversation["id"])
                st.session_state.current_conversation = None
                st.session_state.messages = []
                st.success("Deleted.")
                st.rerun()
            if cols[2].button("🧹 Clear All"):
                store.clear_all()
                st.session_state.current_conversation = None
                st.session_state.messages = []
                st.success("All cleared.")
                st.rerun()
    else:
        st.info("No conversations yet.")

    if st.button("➕ New Conversation"):
        newc = store.create_conversation(name=f"Chat {len(conversations)+1}")
        st.session_state.current_conversation = newc
        st.session_state.messages = []
        st.rerun()

    st.subheader("Backup")
    exp_col, imp_col = st.columns(2)
    if exp_col.button("📤 Export JSON"):
        path = store.export_all("backup.json")
        st.success(f"Exported to {path}")
    imp_file = imp_col.file_uploader("📥 Import JSON", type=["json"])
    if imp_file and st.button("Import now"):
        tmp = "import_backup.json"
        with open(tmp, "wb") as f:
            f.write(imp_file.read())
        ok = store.import_all(tmp)
        if ok:
            st.success("Imported!")
            st.rerun()
        else:
            st.error("Import failed.")

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
    uploaded = st.file_uploader(
        "Upload document (txt, pdf, docx, png, jpg, csv, json)",
        type=["txt","pdf","docx","png","jpg","jpeg","csv","json"]
    )
    col_send, col_voice = st.columns([3,1])
    submitted = col_send.form_submit_button("Send")

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
                unsafe_allow_html=True
            )
    except Exception as e:
        st.error(f"Error generating response: {e}")
        assistant_content = "[Error: Could not generate response]"
    finally:
        elapsed = time.time() - start_time

    response_placeholder.markdown(
        f"<div class='msg assistant'>🤖 {assistant_content}</div>",
        unsafe_allow_html=True
    )

    st.session_state.messages.append({"role": "assistant", "content": assistant_content})
    st.session_state.current_conversation["messages"] = st.session_state.messages
    store.save_conversation(st.session_state.current_conversation)
    store.log_interaction(
        text_in, assistant_content, elapsed,
        conversation_id=st.session_state.current_conversation["id"]
    )

    if st.session_state.get("tts_enabled"):
        speak_text_if_enabled(assistant_content)

    st.rerun()

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("<div class='footer'>GROQ PULSE • Powered by Groq</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)









