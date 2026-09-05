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

Usage:
    python webapp.py
    # or
    uvicorn webapp:app --host 0.0.0.0 --port 5000
"""

import io
import os
import json
import asyncio
import tempfile
import uuid
from datetime import datetime

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
from fastapi.responses import (
    HTMLResponse,
    StreamingResponse,
    JSONResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ============================================================
# CONFIG
# ============================================================

ADMIN_PASSWORD = os.environ.get(
    "ADMIN_PASSWORD", "admin123"
)

WHISPER_SERVER_URL = os.environ.get(
    "WHISPER_SERVER_URL",
    "http://127.0.0.1:8080/inference",
)

# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Government Scheme Assistant",
    docs_url=None,
    redoc_url=None,
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


# ============================================================
# AUTH: Simple session-based admin password
# ============================================================

# In-memory session tokens (good enough for single-user admin)
_admin_tokens = set()


def verify_admin(request: Request):
    """Check if the request has a valid admin token."""
    # Check Authorization header first (from JS fetch)
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and auth[7:] in _admin_tokens:
        return True

    # Fallback to cookie
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
# CHAT API (text input, streaming SSE)
# ============================================================


@app.post("/api/chat")
async def chat(request: Request):
    """
    Accept a text message, return streaming SSE with
    the LLM response.

    Request body: {"message": "user text", "history": [...]}
    """

    body = await request.json()
    user_text = body.get("message", "").strip()
    history = body.get("history", [])

    if not user_text:
        raise HTTPException(
            status_code=400, detail="Empty message"
        )

    llm = get_llm()
    retriever = get_retriever()

    # RAG retrieval
    rag_context = None
    try:
        rag_context, _ = retriever.get_context(user_text)
    except Exception as e:
        print(f"⚠️ RAG error: {e}")

    # Build chat history for the LLM
    chat_history = [{"role": "system", "content": ""}]
    chat_history.extend(history)

    # Language context prefix
    user_message = (
        f"[This request is in English. "
        f"Respond in the same language as the user.]\n\n"
        f"{user_text}"
    )

    async def event_stream():
        full_reply = ""

        for chunk in llm.generate_response(
            user_text=user_message,
            chat_history=chat_history,
            stream=True,
            rag_context=rag_context,
        ):
            full_reply += chunk
            yield f"data: {json.dumps({'text': chunk})}\n\n"

        # Send sources if available
        if rag_context:
            try:
                _, sources = retriever.get_context(
                    user_text
                )
                yield (
                    "data: "
                    + json.dumps(
                        {"sources": sources}
                    )
                    + "\n\n"
                )
            except Exception:
                pass

        yield (
            "data: "
            + json.dumps({"done": True})
            + "\n\n"
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# VOICE API (audio input -> text -> response)
# ============================================================


@app.post("/api/voice")
async def voice_input(
    file: UploadFile = File(...),
):
    """
    Accept audio file, transcribe with whisper,
    then run RAG + LLM. Returns JSON with text
    and optionally audio URL.
    """

    import httpx

    # Read audio
    audio_bytes = await file.read()

    if len(audio_bytes) < 1000:
        raise HTTPException(
            status_code=400,
            detail="Audio too short",
        )

    # Send to whisper.cpp server
    try:
        async with httpx.AsyncClient(
            timeout=300.0
        ) as client:

            response = await client.post(
                WHISPER_SERVER_URL,
                files={
                    "file": (
                        "audio.wav",
                        audio_bytes,
                        "audio/wav",
                    ),
                },
                data={
                    "response_format": "verbose_json",
                    "temperature": "0.0",
                },
            )
            response.raise_for_status()
            result = response.json()
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Whisper server error: {e}",
        )

    # Parse transcription
    text = result.get("text", "").strip()

    if not text:
        return JSONResponse(
            {
                "text": "",
                "reply": "",
                "error": "No speech detected",
            }
        )

    # RAG + LLM
    llm = get_llm()
    retriever = get_retriever()

    rag_context = None
    try:
        rag_context, sources = retriever.get_context(
            text
        )
    except Exception as e:
        print(f"⚠️ RAG error: {e}")
        sources = []

    chat_history = [{"role": "system", "content": ""}]

    user_message = (
        f"[This request is in English. "
        f"Respond in the same language as the user.]\n\n"
        f"{text}"
    )

    reply = ""

    for chunk in llm.generate_response(
        user_text=user_message,
        chat_history=chat_history,
        stream=True,
        rag_context=rag_context,
    ):
        reply += chunk

    return JSONResponse(
        {
            "text": text,
            "reply": reply,
            "sources": sources,
        }
    )


# ============================================================
# TTS API (text -> audio)
# ============================================================


@app.post("/api/tts")
async def text_to_speech(request: Request):
    """Convert text to speech using edge-tts."""

    import edge_tts

    body = await request.json()
    text = body.get("text", "").strip()
    lang = body.get("lang", "en")

    if not text:
        raise HTTPException(
            status_code=400, detail="Empty text"
        )

    voice_map = {
        "en": "en-IN-NeerjaExpressiveNeural",
        "hi": "hi-IN-SwaraNeural",
        "ta": "ta-IN-PallaviNeural",
    }

    voice = voice_map.get(lang, voice_map["en"])

    try:
        communicate = edge_tts.Communicate(text, voice)

        with tempfile.NamedTemporaryFile(
            suffix=".mp3", delete=False
        ) as tmp:
            tmp_path = tmp.name

        await communicate.save(tmp_path)

        with open(tmp_path, "rb") as f:
            audio_data = f.read()

        os.remove(tmp_path)

        return Response(
            content=audio_data,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": (
                    "attachment; filename=speech.mp3"
                ),
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"TTS error: {e}",
        )


# ============================================================
# WEBSOCKET: Voice pipeline (backend owns the mic)
# ============================================================


@app.websocket("/ws/voice")
async def websocket_voice(ws: WebSocket):
    """
    Voice WebSocket — backend owns the microphone.

    Protocol:
      - Browser sends {"action": "start"} → server starts
        capturing mic, processing speech, speaking replies
      - Browser sends {"action": "stop"} → server stops
      - Server sends {"status": "..."} for UI state updates
      - Server sends {"text": "...", "reply": "..."} for
        each conversation turn
      - Server sends binary chunks for TTS audio playback
    """

    import io
    import struct
    import queue

    import numpy as np
    import sounddevice as sd
    import soundfile as sf
    import torch
    import httpx
    import edge_tts

    await ws.accept()
    print("🔌 Voice WebSocket connected")

    # Load Silero VAD
    vad_model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
    )

    # Config
    SAMPLE_RATE = 16000
    VAD_THRESHOLD = 0.5
    SILENCE_PATIENCE_MS = 1200
    IDLE_TIMEOUT_S = 15.0  # auto-stop after this many seconds of silence
    VOLUME_SCALE = 0.1

    audio_q = queue.Queue()
    stop_evt = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    # ---- send helpers (thread-safe) ----

    async def send_json(data):
        await ws.send_json(data)

    async def send_status(status):
        await send_json({"status": status})

    async def send_result(text, reply, sources=None):
        msg = {"text": text, "reply": reply}
        if sources:
            msg["sources"] = sources
        await send_json(msg)

    async def send_tts_audio(text, lang="en"):
        """Generate TTS and send audio chunks."""
        voice_map = {
            "en": "en-IN-NeerjaExpressiveNeural",
            "hi": "hi-IN-SwaraNeural",
            "ta": "ta-IN-PallaviNeural",
        }
        voice = voice_map.get(lang, voice_map["en"])

        try:
            communicate = edge_tts.Communicate(text, voice)

            with tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False
            ) as tmp:
                tmp_path = tmp.name

            await communicate.save(tmp_path)

            data, sr = sf.read(
                tmp_path, dtype="float32"
            )
            os.remove(tmp_path)

            # Convert to int16 PCM and send as chunks
            int16 = (
                np.clip(data * VOLUME_SCALE, -1.0, 1.0)
                * 32767
            ).astype(np.int16)

            # Send header: audio info
            await send_json({
                "status": "tts_audio",
                "sample_rate": sr,
                "channels": 1,
                "total_samples": len(int16),
            })

            # Send audio in chunks
            chunk_size = sr  # 1 second per chunk
            for i in range(0, len(int16), chunk_size):
                chunk = int16[i : i + chunk_size]
                await ws.send_bytes(chunk.tobytes())

            # Signal end of audio
            await send_json({"status": "tts_done"})

        except Exception as e:
            print(f"⚠️ TTS error: {e}")

    # ---- background thread: audio pipeline ----

    def voice_pipeline_thread():
        """
        Runs the voice loop in a background thread.
        Same logic as chatbot.py but without wake word.
        """

        chunk_size = 512
        chunks_per_second = SAMPLE_RATE / chunk_size
        max_silence_chunks = int(
            (SILENCE_PATIENCE_MS / 1000)
            * chunks_per_second
        )

        chat_history = [
            {"role": "system", "content": ""}
        ]

        try:

            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=chunk_size,
                callback=audio_callback,
            ):

                while not stop_evt.is_set():

                    # ---- LISTEN for speech ----

                    recorded = []
                    triggered = False
                    silence_counter = 0
                    listen_start = time.time()

                    while not stop_evt.is_set():

                        # Check idle timeout
                        elapsed = time.time() - listen_start
                        if elapsed > IDLE_TIMEOUT_S:
                            print(
                                f"⏳ Idle timeout "
                                f"({IDLE_TIMEOUT_S}s). "
                                f"Stopping."
                            )
                            asyncio.run(
                                send_status("idle")
                            )
                            return

                        try:
                            chunk = audio_q.get(
                                timeout=0.1
                            )
                        except queue.Empty:
                            continue

                        chunk = chunk.flatten()

                        with torch.no_grad():
                            speech_prob = vad_model(
                                torch.from_numpy(chunk),
                                SAMPLE_RATE,
                            ).item()

                        if speech_prob > VAD_THRESHOLD:
                            if not triggered:
                                triggered = True
                                print(
                                    "  ...speech started"
                                )
                            silence_counter = 0
                        else:
                            if triggered:
                                silence_counter += 1

                        if triggered:
                            recorded.append(chunk)

                            if (
                                silence_counter
                                > max_silence_chunks
                            ):

                                print(
                                    "  ...speech finished"
                                )

                                utterance = (
                                    np.concatenate(
                                        recorded
                                    )
                                )

                                # Run the full pipeline
                                asyncio.run(
                                    process_utterance(
                                        utterance,
                                        chat_history,
                                    )
                                )
                                listen_start = time.time()

                                break

                    # After processing, loop continues
                    # to listen again automatically

        except Exception as e:
            print(f"⚠️ Voice pipeline error: {e}")

        finally:
            print("Voice pipeline thread stopped.")

    async def process_utterance(
        utterance, chat_history
    ):
        """Transcribe → RAG → LLM → TTS → speak."""

        duration = len(utterance) / SAMPLE_RATE
        print(f"🎤 Captured {duration:.1f}s of audio")

        if duration < 0.3:
            return  # too short, skip

        # ---- Whisper transcription ----

        await send_status("transcribing")

        # Build WAV in memory
        wav_buf = io.BytesIO()
        int16 = (
            np.clip(utterance, -1.0, 1.0) * 32767
        ).astype(np.int16)

        data_size = len(int16) * 2
        wav_buf.write(b"RIFF")
        wav_buf.write(
            struct.pack("<I", 36 + data_size)
        )
        wav_buf.write(b"WAVE")
        wav_buf.write(b"fmt ")
        wav_buf.write(struct.pack("<I", 16))
        wav_buf.write(
            struct.pack(
                "<HHIIHH",
                1, 1, SAMPLE_RATE,
                SAMPLE_RATE * 2, 2, 16,
            )
        )
        wav_buf.write(b"data")
        wav_buf.write(struct.pack("<I", data_size))
        wav_buf.write(int16.tobytes())
        wav_buf.seek(0)

        try:
            async with httpx.AsyncClient(
                timeout=300.0
            ) as client:
                resp = await client.post(
                    WHISPER_SERVER_URL,
                    files={
                        "file": (
                            "audio.wav",
                            wav_buf,
                            "audio/wav",
                        ),
                    },
                    data={
                        "response_format": (
                            "verbose_json"
                        ),
                        "temperature": "0.0",
                    },
                )
                resp.raise_for_status()
                result = resp.json()
        except Exception as e:
            print(f"⚠️ Whisper error: {e}")
            return

        text = result.get("text", "").strip()

        if not text:
            print("No speech detected.")
            return

        print(f"📝 Transcribed: {text}")

        # ---- RAG retrieval ----

        await send_status("thinking")

        llm = get_llm()
        retriever = get_retriever()

        rag_context = None
        sources = []
        try:
            rag_context, sources = (
                retriever.get_context(text)
            )
            if rag_context:
                print(
                    f"📚 RAG: {len(sources)} chunks"
                )
        except Exception as e:
            print(f"⚠️ RAG error: {e}")

        # ---- LLM response ----

        user_message = (
            "[This request is in English. "
            "Respond in the same language "
            "as the user.]\n\n"
            f"{text}"
        )

        reply = ""

        for chunk in llm.generate_response(
            user_text=user_message,
            chat_history=chat_history,
            stream=True,
            rag_context=rag_context,
        ):
            reply += chunk

        print(f"🤖 Reply: {reply[:100]}")

        # Send text result to browser
        await send_result(text, reply, sources)

        # ---- TTS → play in browser ----

        await send_status("speaking")
        await send_tts_audio(reply)

        # Done speaking, ready for next utterance
        await send_status("listening")

    # ---- main WebSocket handler ----

    pipeline_thread = None

    try:

        while True:
            msg = await ws.receive()

            if msg.get("type") != "websocket.receive":
                continue

            if msg.get("text"):
                data = json.loads(msg["text"])
                action = data.get("action")

                if action == "start":
                    if (
                        pipeline_thread
                        and pipeline_thread.is_alive()
                    ):
                        continue

                    stop_evt.clear()
                    pipeline_thread = threading.Thread(
                        target=voice_pipeline_thread,
                        daemon=True,
                    )
                    pipeline_thread.start()
                    await send_status("listening")

                elif action == "stop":
                    stop_evt.set()
                    if pipeline_thread:
                        pipeline_thread.join(timeout=2.0)
                    pipeline_thread = None
                    await send_status("idle")

    except WebSocketDisconnect:
        print("🔌 Voice WebSocket disconnected")
        stop_evt.set()
    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")
        stop_evt.set()


# ============================================================
# ADMIN UI
# ============================================================


@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request):
    """Serve the admin document management panel."""

    # Check if already logged in
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

    # Save uploaded file
    upload_dir = os.path.join(
        os.path.dirname(__file__),
        "data",
        "uploads",
    )
    os.makedirs(upload_dir, exist_ok=True)

    filepath = os.path.join(
        upload_dir, file.filename
    )

    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    # Ingest
    from rag.ingest import ingest_document

    store = get_store()

    result = ingest_document(
        filepath,
        vector_store=store,
    )

    return JSONResponse(result)


@app.delete(
    "/admin/api/documents/{doc_id}"
)
async def delete_document(
    doc_id: str,
    request: Request,
    _=Depends(verify_admin),
):
    """Delete a document and all its chunks."""

    store = get_store()
    store.delete_document(doc_id)

    return JSONResponse(
        {"status": "deleted", "doc_id": doc_id}
    )


# ============================================================
# STARTUP
# ============================================================


@app.on_event("startup")
async def startup():
    print(
        "\n========================================"
    )
    print("  Government Scheme RAG Assistant")
    print("========================================")
    print(
        f"  Chat UI:   http://localhost:5000/"
    )
    print(
        f"  Admin:     http://localhost:5000/admin"
    )
    print(
        f"  Password:  {ADMIN_PASSWORD}"
    )
    print(
        "========================================\n"
    )


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
