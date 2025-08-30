# ⚡ GROQ PULSE

GROQ PULSE is a Streamlit AI assistant powered by Groq. Chat via text, voice, or uploaded documents and get real-time AI responses with optional text-to-speech.

---

## Features
- Chat with AI (text & voice)
- Upload documents (`.txt`, `.pdf`, `.docx`, `.csv`, `.json`, `.png`, `.jpg`)
- Conversation management: save, rename, delete
- Backup & restore conversations (JSON)
- Light & dark themes
- Adjustable model parameters (temperature, max tokens, top-p)

---

## Installation
```bash
git clone <repo-url>
cd groq
python -m venv .venv
.\.venv\Scripts\Activate      # Windows
pip install -r requirements.txt
streamlit run app.py
