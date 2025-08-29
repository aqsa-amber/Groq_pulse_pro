# GROQ PULSE (Hardened)
This package contains a Streamlit chat app scaffolded for easy deployment.

Usage:
1. Install dependencies: `pip install -r requirements.txt`
2. (Optional) Set environment variables: GROQ_API_KEY, TTS_PROVIDER
3. Run: `streamlit run app.py`

Notes:
- The included Groq client is a local fallback. If you have a real GROQ API, adapt `utils/groq_client.py` to call it.
- OCR/TTs features require optional dependencies (see requirements.txt).
