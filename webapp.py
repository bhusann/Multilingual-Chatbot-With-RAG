"""
webapp.py
=========
Web UI for the RAG voice assistant.

Routes:
    GET  /           - User chat UI (voice + text)
    POST /api/chat   - Text chat (streaming SSE)
    POST /api/voice  - Voice input (audio -> whisper -> RAG -> LLM -> response)
    GET  /admin      - Admin document management (password-protected)
    POST /admin/api/documents  - List documents
    POST /admin/api/upload     - Upload document
    DELETE /admin/api/documents/{doc_id} - Delete document

    GET  /audio-processor.js   - AudioWorklet script for browser mic capture
    WS   /ws/voice             - Voice pipeline (browser mic -> VAD -> whisper -> LLM -> TTS)

Usage:
    python webapp.py
    # or
    uvicorn webapp:app --host 0.0.0.0 --port 5000
"""

import io
import os
import json
import re
import asyncio
import tempfile
import uuid
import time
import struct
import threading
from datetime import datetime
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import (
    FastAPI,
    Request,
    UploadFile,
    File,
    HTTPException,
    Depends,
    WebSocket,
    WebSocketDisconnect,
)
from starlette.websockets import WebSocketState
from fastapi.responses import (
    HTMLResponse,
    StreamingResponse,
    JSONResponse,
    Response,
)
from fastapi.templating import Jinja2Templates

# ============================================================
# CONFIG
# ============================================================

ADMIN_PASSWORD = os.environ.get(
    "ADMIN_PASSWORD", "admin123"
)

WHISPER_SERVER_URL = os.environ.get(
    "WHISPER_SERVER_URL",
    "http://100.84.186.69:8080/inference",
)

# Voice pipeline config
SAMPLE_RATE = 16000
VAD_CHUNK_SIZE = 512
VAD_THRESHOLD = 0.5
SILENCE_PATIENCE_MS = 1200
IDLE_TIMEOUT_S = 30.0

# ============================================================
# LIFESPAN (Modern FastAPI startup/shutdown)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n========================================")
    print("  Government Scheme RAG Assistant")
    print("========================================")
    print("  Chat UI:   http://localhost:5000/")
    print("  Admin:     http://localhost:5000/admin")
    print(f"  Password:  {ADMIN_PASSWORD}")
    print("========================================\n")

    # Preload embedding model
    from rag.embeddings import get_embedding_model
    print("Preloading embedding model...")
    get_embedding_model()
    print("Embedding model ready.\n")
    yield

# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Government Scheme Assistant",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

# Templates
templates = Jinja2Templates(
    directory=os.path.join(
        os.path.dirname(__file__), "templates"
    )
)

# ============================================================
# LAZY-LOAD SERVICES
# ============================================================

_llm = None
_retriever = None
_store = None
_vad_model = None


def get_llm():
    global _llm
    if _llm is None:
        from llm_service import llm_service

        llm_service.configure_endpoint(
            base_url="http://127.0.0.1:8081/v1",
            model="local",
        )
        _llm = llm_service
    return _llm


def get_retriever():
    global _retriever
    if _retriever is None:
        from rag.retriever import Retriever

        _retriever = Retriever()
    return _retriever


def get_store():
    global _store
    if _store is None:
        from rag.vector_store import VectorStore

        _store = VectorStore()
    return _store


def get_vad_model():
    """Lazy-load Silero VAD model (only when voice is first used)."""
    global _vad_model
    if _vad_model is None:
        import torch
        _vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
        )
        print("Silero VAD loaded for web voice pipeline.")
    return _vad_model


# ============================================================
# SERVER-SIDE CHAT SESSIONS
# ============================================================

_chat_sessions = {}
SESSION_TTL = 3600


def _get_or_create_session(session_id=None):
    """Get an existing session or create a new one."""
    now = time.time()

    # Cleanup stale sessions
    stale = [
        sid for sid, s in _chat_sessions.items()
        if now - s["last_access"] > SESSION_TTL
    ]
    for sid in stale:
        del _chat_sessions[sid]

    if session_id and session_id in _chat_sessions:
        session = _chat_sessions[session_id]
        session["last_access"] = now
        return session_id, session

    # Create new session
    llm = get_llm()
    new_id = uuid.uuid4().hex[:16]
    _chat_sessions[new_id] = {
        "history": [
            {"role": "system", "content": llm.system_prompt}
        ],
        "last_access": now,
    }
    return new_id, _chat_sessions[new_id]


# ============================================================
# AUTH: Simple session-based admin password
# ============================================================

_admin_tokens = set()


def verify_admin(request: Request):
    """Check if the request has a valid admin token."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and auth[7:] in _admin_tokens:
        return True

    token = request.cookies.get("admin_token")
    if token and token in _admin_tokens:
        return True

    raise HTTPException(
        status_code=401, detail="Unauthorized"
    )


# ============================================================
# USER UI
# ============================================================


@app.get("/", response_class=HTMLResponse)
async def chat_ui(request: Request):
    """Serve the user chat interface."""
    return templates.TemplateResponse(
        "index.html", {"request": request}
    )


# ============================================================
# AUDIO WORKLET (served as JS for browser mic capture)
# ============================================================

AUDIO_WORKLET_JS = """
class PCMProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._buffer = new Float32Array(0);
        this._chunkSize = 512;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0] && inputs[0][0];
        if (!input || input.length === 0) return true;

        // Accumulate samples
        const newBuf = new Float32Array(this._buffer.length + input.length);
        newBuf.set(this._buffer);
        newBuf.set(input, this._buffer.length);
        this._buffer = newBuf;

        // Send 512-sample chunks (32ms at 16kHz) as int16
        while (this._buffer.length >= this._chunkSize) {
            const chunk = this._buffer.subarray(0, this._chunkSize);
            const int16 = new Int16Array(this._chunkSize);
            for (let i = 0; i < this._chunkSize; i++) {
                const s = Math.max(-1, Math.min(1, chunk[i]));
                int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            this.port.postMessage(int16.buffer, [int16.buffer]);
            this._buffer = this._buffer.subarray(this._chunkSize);
        }
        return true;
    }
}
registerProcessor('pcm-processor', PCMProcessor);
"""


@app.get("/audio-processor.js")
async def audio_processor_js():
    """Serve the AudioWorklet processor script."""
    return Response(
        content=AUDIO_WORKLET_JS,
        media_type="application/javascript",
        headers={"Cache-Control": "public, max-age=86400"},
    )


# ============================================================
# CHAT API (text input, streaming SSE with server-side sessions)
# ============================================================


@app.post("/api/chat")
async def chat(request: Request):
    """
    Accept a text message, return streaming SSE with the LLM response.
    Uses server-side session for chat history to preserve KV cache across turns.
    """
    body = await request.json()
    user_text = body.get("message", "").strip()
    session_id = body.get("session_id", "")

    if not user_text:
        raise HTTPException(
            status_code=400, detail="Empty message"
        )

    llm = get_llm()
    session_id, session = _get_or_create_session(session_id)
    chat_history = session["history"]

    user_msg = format_user_message(user_text)

    # Use background thread queue to ensure generator doesn't block event loop
    loop = asyncio.get_running_loop()
    q = asyncio.Queue()

    def run_llm():
        try:
            for chunk in llm.generate_response(
                user_text=user_msg,
                chat_history=chat_history,
                stream=True,
            ):
                loop.call_soon_threadsafe(q.put_nowait, ("chunk", chunk))
            loop.call_soon_threadsafe(q.put_nowait, ("done", None))
        except Exception as e:
            loop.call_soon_threadsafe(q.put_nowait, ("error", str(e)))

    threading.Thread(target=run_llm, daemon=True).start()

    async def event_stream():
        while True:
            msg_type, payload = await q.get()
            if msg_type == "chunk":
                yield f"data: {json.dumps({'text': payload})}\n\n"
            elif msg_type == "done":
                yield f"data: {json.dumps({'session_id': session_id})}\n\n"
                yield "data: [DONE]\n\n"
                break
            elif msg_type == "error":
                yield f"data: {json.dumps({'error': payload})}\n\n"
                yield "data: [DONE]\n\n"
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# VOICE API (audio file upload -> text -> response)
# ============================================================


@app.post("/api/voice")
async def voice_input(
    file: UploadFile = File(...),
):
    """
    Accept audio file, transcribe with whisper,
    then run RAG + LLM. Returns JSON with text and reply.
    """
    import httpx

    audio_bytes = await file.read()
    if len(audio_bytes) < 1000:
        raise HTTPException(
            status_code=400,
            detail="Audio too short",
        )

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                WHISPER_SERVER_URL,
                files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                data={"response_format": "verbose_json", "temperature": "0.0"},
            )
            response.raise_for_status()
            result = response.json()
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Whisper server error: {e}",
        )

    text = result.get("text", "").strip()
    if not text:
        return JSONResponse(
            {"text": "", "reply": "", "error": "No speech detected"}
        )

    llm = get_llm()
    _, session = _get_or_create_session()
    chat_history = session["history"]

    # Run LLM in thread to avoid blocking asyncio event loop
    reply = await asyncio.to_thread(
        llm.generate_response,
        user_text=text,
        chat_history=chat_history,
        stream=False,
    )

    return JSONResponse({"text": text, "reply": reply})


# ============================================================
# TTS API (text -> audio)
# ============================================================


@app.post("/api/tts")
async def text_to_speech(request: Request):
    """Convert text to speech using edge-tts."""
    import edge_tts

    body = await request.json()
    text = body.get("text", "").strip()
    lang = body.get("lang")

    if not text:
        raise HTTPException(
            status_code=400, detail="Empty text"
        )

    voice = select_voice(text, preferred_lang=lang)

    try:
        communicate = edge_tts.Communicate(text, voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        await communicate.save(tmp_path)
        with open(tmp_path, "rb") as f:
            audio_data = f.read()
        os.remove(tmp_path)

        return Response(
            content=audio_data,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"TTS error: {e}",
        )


# ============================================================
# WEBSOCKET: Voice pipeline (browser mic -> VAD -> whisper -> LLM -> TTS)
# ============================================================


# ============================================================
# VOICE & MULTILINGUAL TTS
# ============================================================

VOICE_MAP = {
    "en": "en-IN-NeerjaExpressiveNeural",
    "tl": "en-IN-NeerjaExpressiveNeural",
    "hi": "hi-IN-SwaraNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "kn": "kn-IN-SapnaNeural",
    "bn": "bn-IN-TanishaaNeural",
    "mr": "mr-IN-AarohiNeural",
    "gu": "gu-IN-DhwaniNeural",
    "ml": "ml-IN-SobhanaNeural",
    "pa": "pa-IN-VaaniNeural",
}


def detect_script_language(text):
    """Detect Indian language based on Unicode script."""
    for char in text:
        code = ord(char)
        if 0x0900 <= code <= 0x097F:
            return "hi"  # Devanagari (Hindi / Marathi)
        if 0x0980 <= code <= 0x09FF:
            return "bn"  # Bengali
        if 0x0A00 <= code <= 0x0A7F:
            return "pa"  # Gurmukhi (Punjabi)
        if 0x0A80 <= code <= 0x0AFF:
            return "gu"  # Gujarati
        if 0x0B80 <= code <= 0x0BFF:
            return "ta"  # Tamil
        if 0x0C00 <= code <= 0x0C7F:
            return "te"  # Telugu
        if 0x0C80 <= code <= 0x0CFF:
            return "kn"  # Kannada
        if 0x0D00 <= code <= 0x0D7F:
            return "ml"  # Malayalam
    return None


TAMIL_ROMAN_WORDS = {
    "panna", "mudiyuma", "mudiyum", "enna", "epdi", "eppadi", "iruka", "irukinga",
    "irukiya", "panra", "panren", "pannunga", "venum", "vendam", "illai", "illa",
    "aama", "sollunga", "sollu", "inga", "anga", "romba", "nalla", "saptiya",
    "saaptiya", "sapten", "saapten", "theriyuma", "theriyala", "kudunga", "kudu",
    "vaanga", "pora", "poren", "poga", "vandhu", "vantha", "vandha", "irukku",
    "iruku", "yen", "yenga", "engae", "konjam", "seekiram", "ippo", "ippa",
    "naalaikku", "innaikku", "nethu", "enakku", "unakku", "ungalukku", "namma",
    "nanga", "naan", "nee", "neenga",
}

HINDI_ROMAN_WORDS = {
    "kya", "kaise", "kaisa", "kaisi", "hain", "aap", "mujhe", "mujhko", "mera",
    "meri", "mere", "hum", "ham", "karna", "karo", "raha", "rahi", "rahe",
    "chahiye", "nahi", "nahin", "acha", "achha", "accha", "theek", "thik",
    "kyun", "kyon", "kaun", "kab", "kahan", "kidhar", "yeh", "yah", "woh",
    "voh", "mujhse", "aapka", "aapki", "aapke", "pata", "batao", "bataiye",
    "chalo", "dekho", "sakta", "sakti", "sakte",
}


def detect_roman_indian_language(text):
    """Detect Tanglish / Hinglish from Romanized text."""
    text = text.lower().strip()
    words = set(re.findall(r"[a-z]+", text))
    tamil_score = len(words & TAMIL_ROMAN_WORDS)
    hindi_score = len(words & HINDI_ROMAN_WORDS)

    if tamil_score >= 2 and tamil_score > hindi_score:
        return "tl"
    if hindi_score >= 2 and hindi_score > tamil_score:
        return "hi"
    return None


LANG_NAMES = {
    "en": "English",
    "tl": "Tanglish",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "ml": "Malayalam",
    "pa": "Punjabi",
}


def format_user_message(user_text, whisper_lang="en"):
    """
    Format the user message with per-turn language instruction tag.
    Keeps system prompt static (at index 0) for KV cache preservation.
    """
    lang = detect_script_language(user_text)
    if not lang:
        lang = detect_roman_indian_language(user_text)
    if not lang and whisper_lang in VOICE_MAP:
        lang = whisper_lang
    if not lang:
        lang = "en"

    lang_name = LANG_NAMES.get(lang, "English")
    tanglish_note = ""
    if lang == "tl":
        tanglish_note = (
            " Tanglish means Tamil written in Latin script "
            "mixed with English words (e.g. 'Apply panna mudiyum', "
            "'konjam wait pannungo'). Do NOT use Tamil script."
        )

    return (
        f"[This request is in {lang_name}. "
        f"Respond in {lang_name}. "
        f"If the user has explicitly asked to speak "
        f"in a different language in this conversation, "
        f"follow that instruction instead.{tanglish_note}]\n\n"
        f"{user_text}"
    )


def select_voice(text, preferred_lang=None):
    """Select the best Edge-TTS neural voice for the given text."""
    if preferred_lang and preferred_lang in VOICE_MAP:
        return VOICE_MAP[preferred_lang]

    # Detect by native Unicode script
    lang = detect_script_language(text)
    if not lang:
        # Check romanized Indian language (Tanglish / Hinglish)
        lang = detect_roman_indian_language(text)

    return VOICE_MAP.get(lang, VOICE_MAP["en"])


def _clean_for_tts(t):
    """Clean markdown formatting for TTS."""
    if not t:
        return t
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"\*(.+?)\*", r"\1", t)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"^#{1,6}\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"\s{2,}", " ", t)
    t = t.replace("**", "").replace("__", "")
    return t.strip()


@app.websocket("/ws/voice")
async def websocket_voice(ws: WebSocket):
    """
    Voice WebSocket — browser captures mic and streams audio.

    States:
      - 'idle': Voice not running
      - 'listening': Receiving mic frames, running VAD
      - 'processing': Transcribing with whisper, running LLM
      - 'speaking': Streaming TTS audio to browser
      - 'waiting_playback': Browser is playing TTS, mic muted
    """
    import numpy as np
    import torch
    import httpx
    import edge_tts

    await ws.accept()
    print("🔌 Voice WebSocket connected")

    vad_model = get_vad_model()

    # Per-connection state
    state = "idle"  # idle | listening | processing | speaking | waiting_playback
    vad_buffer = np.zeros(0, dtype=np.float32)
    recorded_chunks = []
    speech_triggered = False
    silence_counter = 0
    last_speech_time = time.time()

    chunks_per_second = SAMPLE_RATE / VAD_CHUNK_SIZE
    max_silence_chunks = int(
        (SILENCE_PATIENCE_MS / 1000) * chunks_per_second
    )

    # Server-side chat session shared between voice and text chat
    session_id = ws.query_params.get("session_id")
    session_id, session = _get_or_create_session(session_id)
    chat_history = session["history"]
    llm = get_llm()

    async def safe_send_json(payload):
        if ws.client_state == WebSocketState.CONNECTED:
            try:
                await ws.send_json(payload)
                return True
            except Exception:
                return False
        return False

    async def safe_send_bytes(payload):
        if ws.client_state == WebSocketState.CONNECTED:
            try:
                await ws.send_bytes(payload)
                return True
            except Exception:
                return False
        return False

    async def handle_utterance(utterance):
        nonlocal state, last_speech_time

        duration = len(utterance) / SAMPLE_RATE
        print(f"🎤 Captured {duration:.1f}s of audio")

        if duration < 0.3:
            state = "listening"
            last_speech_time = time.time()
            await safe_send_json({"status": "listening"})
            return

        if ws.client_state != WebSocketState.CONNECTED:
            return

        # 1. Transcribing
        state = "processing"
        await safe_send_json({"status": "transcribing"})

        int16 = (np.clip(utterance, -1.0, 1.0) * 32767).astype(np.int16)
        wav_buf = io.BytesIO()
        data_size = len(int16) * 2
        wav_buf.write(b"RIFF")
        wav_buf.write(struct.pack("<I", 36 + data_size))
        wav_buf.write(b"WAVE")
        wav_buf.write(b"fmt ")
        wav_buf.write(struct.pack("<I", 16))
        wav_buf.write(struct.pack("<HHIIHH", 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16))
        wav_buf.write(b"data")
        wav_buf.write(struct.pack("<I", data_size))
        wav_buf.write(int16.tobytes())
        wav_buf.seek(0)

        text = ""
        whisper_lang = "en"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    WHISPER_SERVER_URL,
                    files={"file": ("audio.wav", wav_buf, "audio/wav")},
                    data={"response_format": "verbose_json", "temperature": "0.0"},
                )
                resp.raise_for_status()
                result = resp.json()
                text = result.get("text", "").strip()
                whisper_lang = str(result.get("language", "en")).lower()
        except Exception as e:
            print(f"⚠️ Whisper error: {e}")
            state = "listening"
            last_speech_time = time.time()
            await safe_send_json({"status": "listening"})
            return

        if not text:
            print("No speech detected.")
            state = "listening"
            last_speech_time = time.time()
            await safe_send_json({"status": "listening"})
            return

        print(f"📝 Transcribed: {text}")

        if ws.client_state != WebSocketState.CONNECTED:
            return

        # 2. LLM response (in worker thread to NEVER block asyncio keepalive)
        await safe_send_json({"status": "thinking"})

        user_msg = format_user_message(text, whisper_lang=whisper_lang)

        try:
            reply = await asyncio.to_thread(
                llm.generate_response,
                user_text=user_msg,
                chat_history=chat_history,
                stream=False,
            )
        except Exception as e:
            print(f"⚠️ LLM error: {e}")
            reply = "I encountered an error generating the response."

        print(f"🤖 Reply: {reply[:100]}...")

        if ws.client_state != WebSocketState.CONNECTED:
            return

        await safe_send_json({"text": text, "reply": reply})

        # 3. TTS response
        state = "speaking"
        await safe_send_json({"status": "speaking"})

        tts_text = _clean_for_tts(reply)
        has_audio = False

        if tts_text and ws.client_state == WebSocketState.CONNECTED:
            voice = select_voice(tts_text)
            print(f"🔊 Synthesizing TTS with voice: {voice}")
            try:
                communicate = edge_tts.Communicate(tts_text, voice)
                await safe_send_json({"status": "tts_start"})

                async for tts_chunk in communicate.stream():
                    if ws.client_state != WebSocketState.CONNECTED:
                        break
                    if tts_chunk["type"] == "audio":
                        if await safe_send_bytes(tts_chunk["data"]):
                            has_audio = True

                await safe_send_json({"status": "tts_done"})
            except Exception as e:
                print(f"⚠️ TTS error: {e}")

        if ws.client_state != WebSocketState.CONNECTED:
            return

        if has_audio:
            state = "waiting_playback"
            # Set a fallback timer in case client fails to send "ready"
            asyncio.create_task(_playback_timeout_fallback())
        else:
            state = "listening"
            last_speech_time = time.time()
            await safe_send_json({"status": "listening"})

    async def _playback_timeout_fallback():
        nonlocal state, last_speech_time, vad_buffer, recorded_chunks, speech_triggered, silence_counter
        await asyncio.sleep(25.0)  # Max playback time
        if ws.client_state != WebSocketState.CONNECTED:
            return
        if state == "waiting_playback":
            print("⏳ Playback timeout fallback, resuming listening")
            vad_buffer = np.zeros(0, dtype=np.float32)
            recorded_chunks.clear()
            speech_triggered = False
            silence_counter = 0
            vad_model.reset_states()
            last_speech_time = time.time()
            state = "listening"
            await safe_send_json({"status": "listening"})

    try:
        while True:
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                print("🔌 Voice WebSocket disconnected")
                break
            except RuntimeError as e:
                if "disconnect message has been received" in str(e):
                    print("🔌 Voice WebSocket closed (disconnect message received)")
                    break
                raise

            msg_type = msg.get("type")

            if msg_type == "websocket.disconnect":
                print("🔌 Voice WebSocket received disconnect")
                break

            if msg_type not in ("websocket.receive",):
                continue

            # Binary: audio data from browser mic
            if msg.get("bytes"):
                # ONLY process audio when actively in "listening" state
                if state != "listening":
                    continue

                raw = msg["bytes"]
                int16 = np.frombuffer(raw, dtype=np.int16)
                float32 = int16.astype(np.float32) / 32768.0

                vad_buffer = np.concatenate([vad_buffer, float32])

                while len(vad_buffer) >= VAD_CHUNK_SIZE:
                    chunk = vad_buffer[:VAD_CHUNK_SIZE]
                    vad_buffer = vad_buffer[VAD_CHUNK_SIZE:]

                    with torch.no_grad():
                        speech_prob = vad_model(
                            torch.from_numpy(chunk),
                            SAMPLE_RATE,
                        ).item()

                    if speech_prob > VAD_THRESHOLD:
                        if not speech_triggered:
                            speech_triggered = True
                            print("  ...speech started")
                        silence_counter = 0
                    else:
                        if speech_triggered:
                            silence_counter += 1

                    if speech_triggered:
                        recorded_chunks.append(chunk)

                        if silence_counter > max_silence_chunks:
                            print("  ...speech finished")
                            utterance = np.concatenate(recorded_chunks)

                            # Immediately switch state so subsequent frames are dropped
                            state = "processing"
                            recorded_chunks.clear()
                            vad_buffer = np.zeros(0, dtype=np.float32)
                            speech_triggered = False
                            silence_counter = 0

                            # Run processing task asynchronously
                            asyncio.create_task(handle_utterance(utterance))
                            break

                # Idle timeout check (ONLY when in listening mode and no active speech)
                if state == "listening" and not speech_triggered:
                    if time.time() - last_speech_time > IDLE_TIMEOUT_S:
                        print(f"⏳ Idle timeout ({IDLE_TIMEOUT_S}s)")
                        state = "idle"
                        vad_buffer = np.zeros(0, dtype=np.float32)
                        recorded_chunks.clear()
                        speech_triggered = False
                        silence_counter = 0
                        await safe_send_json({"status": "idle"})

                continue

            # Text: JSON command from browser
            if msg.get("text"):
                data = json.loads(msg["text"])
                action = data.get("action")

                if action == "start":
                    client_sid = data.get("session_id")
                    if client_sid:
                        session_id, session = _get_or_create_session(client_sid)
                        chat_history = session["history"]

                    state = "listening"
                    vad_buffer = np.zeros(0, dtype=np.float32)
                    recorded_chunks.clear()
                    speech_triggered = False
                    silence_counter = 0
                    last_speech_time = time.time()
                    vad_model.reset_states()

                    await safe_send_json({"status": "listening", "session_id": session_id})
                    print(f"🎙️ Voice session started (listening for user, session: {session_id})")

                elif action == "stop":
                    if state != "idle":
                        state = "idle"
                        vad_buffer = np.zeros(0, dtype=np.float32)
                        recorded_chunks.clear()
                        speech_triggered = False
                        silence_counter = 0
                        await safe_send_json({"status": "idle"})
                        print("⏹️ Voice session stopped")

                elif action in ("ready", "playback_done"):
                    # Browser finished playing TTS audio; resume listening
                    if state in ("waiting_playback", "speaking"):
                        state = "listening"
                        vad_buffer = np.zeros(0, dtype=np.float32)
                        recorded_chunks.clear()
                        speech_triggered = False
                        silence_counter = 0
                        last_speech_time = time.time()
                        vad_model.reset_states()
                        await safe_send_json({"status": "listening"})
                        print("🎧 Playback finished, resumed listening")

    except WebSocketDisconnect:
        print("🔌 Voice WebSocket disconnected")
    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")


# ============================================================
# ADMIN UI
# ============================================================


@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request):
    """Serve the admin document management panel."""
    token = request.cookies.get("admin_token")
    logged_in = token and token in _admin_tokens

    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "logged_in": logged_in,
        },
    )


@app.post("/admin/login")
async def admin_login(request: Request):
    """Authenticate admin with password."""
    body = await request.json()
    password = body.get("password", "")

    if password != ADMIN_PASSWORD:
        raise HTTPException(
            status_code=401,
            detail="Wrong password",
        )

    token = uuid.uuid4().hex
    _admin_tokens.add(token)

    return JSONResponse(
        {"token": token, "status": "ok"}
    )


@app.post("/admin/api/documents")
async def list_documents(
    request: Request,
    _=Depends(verify_admin),
):
    """List all documents in the vector store."""
    store = get_store()
    docs = store.list_documents()
    return JSONResponse({"documents": docs})


@app.post("/admin/api/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    _=Depends(verify_admin),
):
    """Upload and ingest a document (PDF/TXT)."""
    upload_dir = os.path.join(
        os.path.dirname(__file__),
        "data",
        "uploads",
    )
    os.makedirs(upload_dir, exist_ok=True)

    filepath = os.path.join(upload_dir, file.filename)
    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    from rag.ingest import ingest_document
    store = get_store()
    result = ingest_document(filepath, vector_store=store)
    return JSONResponse(result)


@app.delete("/admin/api/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    request: Request,
    _=Depends(verify_admin),
):
    """Delete a document and all its chunks."""
    store = get_store()
    store.delete_document(doc_id)
    return JSONResponse({"status": "deleted", "doc_id": doc_id})


# ============================================================
# ADMIN: GRIEVANCE TICKETS
# ============================================================


@app.post("/admin/api/grievances")
async def list_grievances(
    request: Request,
    _=Depends(verify_admin),
):
    """List all grievance tickets, newest first."""
    from grievance_store import get_grievance_store

    store = get_grievance_store()
    tickets = store.list_tickets()
    return JSONResponse({"tickets": tickets})


@app.patch("/admin/api/grievances/{ticket_id}")
async def update_grievance_status(
    ticket_id: str,
    request: Request,
    _=Depends(verify_admin),
):
    """Update a ticket's status (staff only)."""
    from grievance_store import get_grievance_store

    body = await request.json()
    status = body.get("status", "")

    store = get_grievance_store()
    try:
        ticket = store.update_status(ticket_id, status)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if ticket is None:
        raise HTTPException(
            status_code=404, detail="Ticket not found"
        )

    return JSONResponse({"status": "updated", "ticket": ticket})


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "webapp:app",
        host="0.0.0.0",
        port=5000,
        reload=False,
    )
