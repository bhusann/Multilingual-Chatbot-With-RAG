# Multilingual Voice Assistant

Voice-to-voice multilingual assistant for government schemes. Speech is
transcribed via whisper.cpp server, the LLM translates intent to English,
searches the live web via DuckDuckGo (as many rounds as needed), and
replies in the user's language (Tanglish, Tamil, Hindi, English, etc.),
spoken aloud with edge-tts.

## Features

- Hands-free wake-word control: say **"alexa"** to start a conversation,
  and say **"alexa"** again to interrupt while speaking or searching
- Continuous conversation: after a reply it keeps listening, only
  returning to the wake-word idle state after ~15s of inactivity
- Speech input via microphone with Silero VAD
- Whisper.cpp server for transcription, language auto-detection
- LLM agent loop with `web_search` tool calling (OpenAI-compatible)
- Parallel web searches: multiple tool calls per round run concurrently via
  `ThreadPoolExecutor` (up to `MAX_PARALLEL_SEARCHES=4`)
- Streaming responses for the final answer
- Custom LLM endpoint support (local llama.cpp, API models, etc.)
- Always searches for the latest data (current year is injected into prompts)
- Language mirroring: Tanglish -> Tanglish, Tamil -> Tamil script, etc.
- Soft acceptance chime when the wake word is detected / interrupt is accepted
- edge-tts spoken replies using natural Indian voices

## Architecture

- **whisper.cpp** — runs as a separate server on port 8080 for speech-to-text
- **llama.cpp** — runs as a separate server on port 8081 for LLM inference
- **OpenCode Zen** — cloud API fallback (set `OPENCODE_API_KEY` in `.env`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your key (optional for local LLM):

```bash
cp .env.example .env
# edit .env -> OPENCODE_API_KEY=your-key-here  (optional)
```

## Run

Start whisper.cpp server:
```bash
LD_LIBRARY_PATH=~/programs/sandyproj/whisper-bin-ubuntu-x64 \
~/programs/sandyproj/whisper-bin-ubuntu-x64/whisper-server \
  --host 127.0.0.1 --port 8080 \
  -m ~/programs/sandyproj/ggml-large-v3-q5_0.bin -l auto
```

Start llama.cpp server (or any OpenAI-compatible endpoint):
```bash
# Your llama.cpp server on port 8081
```

Run the chatbot:
```bash
python chatbot.py
```

Say **"alexa"** to wake the assistant, then ask your question.
Say **"alexa"** at any time (while the reply is playing or the search
is running) to stop it and speak again.

Say "talk in Tamil" / "switch to Hindi" / "speak in tanglish" to change
the response language.

## Files

- `chatbot.py` — mic capture, VAD, wake word, whisper.cpp client, language resolution, TTS
- `llm_service.py` — search-agent LLM service with streaming, custom endpoint support

## Configuration

Environment variables (`.env`):
- `OPENCODE_API_KEY` — API key for OpenCode Zen (optional for local LLM)
- `OPENCODE_MODEL` — model name (default: `deepseek-v4-flash-free`)
- `WHISPER_SERVER_URL` — whisper.cpp server URL (default: `http://127.0.0.1:8080/inference`)

The LLM endpoint defaults to `http://127.0.0.1:8081/v1` (local llama.cpp).
Override with `llm_service.configure_endpoint()` in `chatbot.py`.
