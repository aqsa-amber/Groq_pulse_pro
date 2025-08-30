import os
import threading

def speak_text_if_enabled(text: str) -> bool:
    """
    Speak text aloud if TTS is enabled.
    Controlled via env or Streamlit secrets: TTS_PROVIDER=pyttsx3 (optional: TTS_RATE, TTS_VOLUME, TTS_VOICE).
    Runs asynchronously; returns True if speech started, else False.
    """
    provider = os.getenv("TTS_PROVIDER", "").lower().strip()
    if not provider:
        return False  # disabled

    if provider == "pyttsx3":
        def _run_tts(text: str):
            try:
                import pyttsx3
                engine = pyttsx3.init()

                rate = int(os.getenv("TTS_RATE", "175"))         # words per minute
                volume = float(os.getenv("TTS_VOLUME", "1.0"))   # 0.0 - 1.0
                voice = os.getenv("TTS_VOICE", "")               # substring of voice name

                engine.setProperty("rate", rate)
                engine.setProperty("volume", volume)

                if voice:
                    voices = engine.getProperty("voices")
                    for v in voices:
                        if voice.lower() in v.name.lower():
                            engine.setProperty("voice", v.id)
                            break

                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[TTS Error] {e}")

        threading.Thread(target=_run_tts, args=(text,), daemon=True).start()
        return True

    return False

