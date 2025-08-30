import os
import time
import json
import requests
from dataclasses import dataclass
from typing import Generator, Any, Dict

@dataclass
class StreamChunk:
    """Represents a chunk of streamed text from Groq API."""
    text: str

class GroqClient:
    """
    GroqClient connects to the Groq API for chat completions.

    Usage:
        import streamlit as st
        client = GroqClient(api_key=st.secrets["GROQ_API_KEY"])
        for chunk in client.stream_response("Hello!"):
            print(chunk.text, end="")
    """
    def __init__(self, api_key: str = None, api_url: str = None):
        """
        Initialize GroqClient with API key and optional API URL.

        Args:
            api_key (str, optional): Groq API key. Defaults to environment or Streamlit secret.
            api_url (str, optional): Custom API URL. Defaults to official Groq API.
        """
        # Load API key from argument, environment, or Streamlit secrets
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.api_url = api_url or os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")

        if not self.api_key:
            print("⚠️ Warning: No API key provided. GroqClient will run in local fallback mode.")

    def stream_response(self, prompt: str, params: Dict[str, Any] = None) -> Generator[StreamChunk, None, None]:
        """
        Stream a response from Groq API for the given prompt.

        Args:
            prompt (str): User input prompt.
            params (Dict[str, Any], optional): Model parameters like temperature, max_tokens, top_p.

        Yields:
            StreamChunk: Chunk of generated text.
        """
        # Default model parameters
        params = params or {
            "model": "llama3-8b-8192",
            "temperature": 0.2,
            "max_tokens": 512,
            "top_p": 1.0
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": params.get("model", "llama3-8b-8192"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(params.get("temperature", 0.2)),
            "top_p": float(params.get("top_p", 1.0)),
            "max_completion_tokens": int(params.get("max_tokens", 512)),
            "stream": True
        }

        # Local fallback if no API key provided
        if not self.api_key:
            full_response = "[LOCAL FALLBACK]\n\nResponse to: " + prompt
            for i in range(0, len(full_response), 60):
                time.sleep(0.02)
                yield StreamChunk(text=full_response[i:i+60])
            return

        # Attempt to stream from Groq API
        try:
            with requests.post(self.api_url, headers=headers, json=payload, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if line.startswith(b"data: "):
                        chunk_data = line[len(b"data: "):].decode("utf-8")
                        if chunk_data.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(chunk_data)
                            delta = data["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield StreamChunk(text=delta)
                        except Exception:
                            # Ignore malformed SSE chunks
                            continue
        except Exception as e:
            # If API call fails, yield an error message as fallback
            yield StreamChunk(text=f"[Error: {e}]")

