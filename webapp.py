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
# WEBSOCKET: Real-time voice streaming
# ============================================================


@app.websocket("/ws/voice")
async def websocket_voice(ws: WebSocket):
    """
    WebSocket for real-time voice streaming.

    Protocol:
      - Client opens connection
      - Client sends binary messages: raw float32 PCM samples
        (16kHz, mono, 4-byte float32 per sample)
      - Client sends text message "done" when utterance ends
      - Server transcribes, runs RAG+LLM, sends JSON response
      - Server sends {"status": "ready"} when ready for next utterance
    """

    await ws.accept()
    print("🔌 Voice WebSocket connected")

    import httpx
    import numpy as np

    audio_chunks = []

    try:
        while True:
            msg = await ws.receive()

            if msg.get("type") == "websocket.receive":

                # Binary message: raw PCM audio chunk
                if msg.get("bytes"):
                    audio_chunks.append(msg["bytes"])

                # Text message: control commands
                elif msg.get("text"):

                    text = msg["text"].strip()

                    if text == "done":
                        # Process the complete utterance
                        if not audio_chunks:
                            await ws.send_json(
                                {"error": "No audio captured"}
                            )
                            continue

                        # Concatenate all chunks into one buffer
                        full_audio = b"".join(audio_chunks)
                        audio_chunks = []

                        # Convert float32 PCM to WAV
                        samples = np.frombuffer(
                            full_audio, dtype=np.float32
                        )

                        duration = (
                            len(samples) / 16000
                        )
                        print(
                            f"🎤 Captured {duration:.1f}s "
                            f"of audio"
                        )

                        if duration < 0.3:
                            await ws.send_json(
                                {
                                    "error": (
                                        "Audio too short. "
                                        "Speak for at least "
                                        "1 second."
                                    )
                                }
                            )
                            continue

                        # Build WAV in memory
                        import struct
                        import io

                        wav_buf = io.BytesIO()
                        int16 = (
                            np.clip(samples, -1.0, 1.0)
                            * 32767
                        ).astype(np.int16)

                        # WAV header
                        data_size = len(int16) * 2
                        wav_buf.write(b"RIFF")
                        wav_buf.write(
                            struct.pack(
                                "<I",
                                36 + data_size,
                            )
                        )
                        wav_buf.write(b"WAVE")
                        wav_buf.write(b"fmt ")
                        wav_buf.write(
                            struct.pack("<I", 16)
                        )
                        wav_buf.write(
                            struct.pack(
                                "<HHIIHH",
                                1, 1, 16000,
                                32000, 2, 16,
                            )
                        )
                        wav_buf.write(b"data")
                        wav_buf.write(
                            struct.pack("<I", data_size)
                        )
                        wav_buf.write(int16.tobytes())
                        wav_buf.seek(0)

                        # Send to whisper
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
                                        "temperature": (
                                            "0.0"
                                        ),
                                    },
                                )
                                resp.raise_for_status()
                                result = resp.json()
                        except Exception as e:
                            await ws.send_json(
                                {
                                    "error": (
                                        f"Whisper error: {e}"
                                    )
                                }
                            )
                            continue

                        transcription = result.get(
                            "text", ""
                        ).strip()

                        if not transcription:
                            await ws.send_json(
                                {
                                    "text": "",
                                    "reply": "",
                                    "error": (
                                        "No speech detected"
                                    ),
                                }
                            )
                            continue

                        print(
                            f"📝 Transcribed: "
                            f"{transcription}"
                        )

                        # RAG retrieval
                        llm = get_llm()
                        retriever = get_retriever()

                        rag_context = None
                        sources = []
                        try:
                            rag_context, sources = (
                                retriever.get_context(
                                    transcription
                                )
                            )
                        except Exception as e:
                            print(f"⚠️ RAG error: {e}")

                        # LLM response
                        chat_history = [
                            {
                                "role": "system",
                                "content": "",
                            }
                        ]

                        user_message = (
                            "[This request is in English. "
                            "Respond in the same language "
                            "as the user.]\n\n"
                            f"{transcription}"
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

                        # Send response
                        await ws.send_json(
                            {
                                "text": transcription,
                                "reply": reply,
                                "sources": sources,
                            }
                        )

                        # Signal client to start listening again
                        await ws.send_json(
                            {"status": "listen"}
                        )

                    elif text == "ping":
                        await ws.send_json(
                            {"status": "ready"}
                        )

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
