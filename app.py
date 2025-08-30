import os
import time
import json
import queue
import threading
import streamlit as st
import av
import numpy as np

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

st.write("API Key Loaded:", (api_key[:5] + "…") if api_key else "None")

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
# WebRTC (browser mic support) - Vosk STT
# -------------------------------------------------
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
from vosk import Model, KaldiRecognizer

# Try to import scipy for high-quality resampling; fall back if unavailable
try:
    import scipy.signal as _scipy_signal  # optional
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Load Vosk model once
if "vosk_model" not in st.session_state:
    model_path = "vosk-model-small-en-us-0.15"  # make sure this folder exists in project root
    if not os.path.exists(model_path):
        st.error(f"❌ Vosk model not found at '{model_path}'. "
                 f"Download & unzip a model (e.g., vosk-model-small-en-us-0.15) into project root.")
        st.stop()
    st.session_state.vosk_model = Model(model_path)
vosk_model = st.session_state.vosk_model

def _to_mono_float32(arr: np.ndarray) -> np.ndarray:
    """
    Normalize AV frame ndarray to mono float32 in [-1, 1].
    Handles shapes: (samples,), (samples, channels), (channels, samples)
    """
    if arr.ndim == 1:
        mono = arr
    elif arr.ndim == 2:
        # Heuristic: if first dim looks like channels (<=8), transpose
        if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
            arr = arr.T
        mono = arr.mean(axis=1)
    else:
        mono = arr.flatten()

    if mono.dtype != np.float32:
        mono = mono.astype(np.float32, copy=False)

    # If samples appear to be int16 scaled, normalize to [-1,1]
    # Many browsers already give float32 in [-1,1]; this is safe anyway.
    np.clip(mono, -1.0, 1.0, out=mono)
    return mono

def _resample_to_16k(mono_f32: np.ndarray, input_rate: int) -> np.ndarray:
    """
    Resample mono float32 audio to 16kHz float32.
    Uses scipy.signal.resample_poly if available; otherwise falls back to numpy interpolation.
    """
    target_rate = 16000
    if input_rate == target_rate:
        return mono_f32

    if _HAS_SCIPY:
        # High quality polyphase resampling
        return _scipy_signal.resample_poly(mono_f32, target_rate, input_rate).astype(np.float32, copy=False)

    # Lightweight fallback: linear interpolation
    duration = len(mono_f32) / float(input_rate)
    new_len = int(round(duration * target_rate))
    if new_len <= 1:
        return np.zeros(0, dtype=np.float32)
    old_times = np.linspace(0.0, duration, num=len(mono_f32), endpoint=False)
    new_times = np.linspace(0.0, duration, num=new_len, endpoint=False)
    return np.interp(new_times, old_times, mono_f32).astype(np.float32, copy=False)

def _float32_to_int16_pcm(mono_f32_16k: np.ndarray) -> bytes:
    """
    Convert mono float32 [-1,1] @16kHz to int16 PCM bytes.
    """
    np.clip(mono_f32_16k, -1.0, 1.0, out=mono_f32_16k)
    pcm_i16 = (mono_f32_16k * 32767.0).astype(np.int16)
    return pcm_i16.tobytes()

class AudioProcessor(AudioProcessorBase):
    """
    Non-blocking audio processor:
    - recv_audio_frame(): very fast — convert/resample & enqueue
    - background thread: reads queue, feeds Vosk, posts results
    This prevents WebRTC from buffering/stopping.
    """
    def __init__(self):
        self.q: "queue.Queue[bytes]" = queue.Queue(maxsize=50)
        self.recognizer = KaldiRecognizer(vosk_model, 16000)
        self.running = True

        # Start background recognizer thread
        self.thread = threading.Thread(target=self._process_audio_loop, daemon=True)
        self.thread.start()

    def recv_audio_frame(self, frame: av.AudioFrame):
        # Convert AV frame -> mono float32
        arr = frame.to_ndarray()
        mono_f32 = _to_mono_float32(arr)

        # Resample to 16kHz
        in_rate = int(getattr(frame, "sample_rate", 48000) or 48000)
        mono_16k = _resample_to_16k(mono_f32, in_rate)

        # Convert to PCM16 bytes
        pcm_bytes = _float32_to_int16_pcm(mono_16k)

        # Enqueue without blocking (drop if queue is full to keep real-time)
        try:
            self.q.put_nowait(pcm_bytes)
        except queue.Full:
            pass  # drop chunk; better to drop than block

        return frame

    def _process_audio_loop(self):
        # Accumulate small chunks to form reasonable buffers for Vosk
        buf = bytearray()
        min_bytes = int(0.4 * 16000 * 2)  # ~0.4s @ 16kHz, 16-bit

        while self.running:
            try:
                chunk = self.q.get(timeout=0.5)
            except queue.Empty:
                # If we have partial audio, try processing it
                if buf:
                    self._feed_recognizer(bytes(buf))
                    buf.clear()
                continue

            buf.extend(chunk)

            if len(buf) >= min_bytes:
                self._feed_recognizer(bytes(buf))
                buf.clear()

    def _feed_recognizer(self, data: bytes):
        # Feed a chunk into Vosk and handle results
        try:
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = (result.get("text") or "").strip()
                if text:
                    st.session_state["voice_prompt"] = text
                    if st.session_state.get("debug_mode"):
                        st.write("✅ Full result:", result)
            else:
                partial = json.loads(self.recognizer.PartialResult())
                part = (partial.get("partial") or "").strip()
                if part:
                    st.session_state["voice_prompt"] = part
                    if st.session_state.get("debug_mode"):
                        st.write("⏳ Partial:", partial)
        except Exception as e:
            if st.session_state.get("debug_mode"):
                st.warning(f"Recognizer error: {e}")

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
        "Theme",
        ["light", "dark"],
        index=0 if st.session_state.theme == "light" else 1
    )

    st.subheader("Model Parameters")
    st.session_state.settings["model"] = st.selectbox(
        "Model",
        ["llama3-8b-8192", "llama3-70b-8192"],
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
    st.session_state.debug_mode = st.toggle("Enable Debug Logs", value=False)

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
if submitted and text_in and text_in.strip():
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
    store.log_interaction(text_in, assistant_content, elapsed, conversation_id=st.session_state.current_conversation["id"])

    if st.session_state.get("tts_enabled"):
        speak_text_if_enabled(assistant_content)

    st.rerun()

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("<div class='footer'>GROQ PULSE • Powered by Groq</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)








