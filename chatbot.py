import asyncio
import io
import json
import queue
import tempfile
import os
import sys
import re
import threading
import time

from huggingface_hub.utils import enable_progress_bars
enable_progress_bars()

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

import edge_tts

from openwakeword import Model as WakeWordModel

import openwakeword as _oww_pkg

from llm_service import llm_service


# ============================================================
# RAG RETRIEVER
# ============================================================

_rag_retriever = None


def get_rag_retriever():
    """Lazy-load the RAG retriever (avoids loading
    embedding model at startup if RAG is not used)."""

    global _rag_retriever

    if _rag_retriever is None:
        from rag.retriever import Retriever
        _rag_retriever = Retriever()

    return _rag_retriever


# ============================================================
# CONFIG
# ============================================================

SAMPLE_RATE = 16000

VAD_THRESHOLD = 0.5
SILENCE_PATIENCE_MS = 1000

VOLUME_SCALE = 0.1

# Wake word / interrupt detection (openWakeWord)
WAKE_WORD_MODEL = "alexa_v0.1"
WAKE_THRESHOLD = 0.5
WAKE_COOLDOWN_S = 1.0

# How long (seconds) of no wake-word activity before the
# mic-stream is considered silent/idle again.
WAKE_IDLE_TIMEOUT_S = 15.0


# ============================================================
# LANGUAGE / VOICE CONFIG
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

LANG_NAME_TO_CODE = {
    "english": "en",
    "tanglish": "tl",
    "thanglish": "tl",
    "hindi": "hi",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "bengali": "bn",
    "marathi": "mr",
    "gujarati": "gu",
    "malayalam": "ml",
    "punjabi": "pa",
}

DEFAULT_VOICE = "en-IN-NeerjaExpressiveNeural"


# ============================================================
# WHISPER HALLUCINATION FILTER
# ============================================================

HALLUCINATION_PHRASES = [
    "thank you very much",
    "thanks for watching",
    "please subscribe",
    "thank you",
    "watching",
]


# ============================================================
# NATIVE SCRIPT LANGUAGE DETECTION
# ============================================================

def detect_script_language(text):
    """
    Detect Indian languages based on Unicode script.

    This is more reliable than langdetect for short text.
    """

    for char in text:

        code = ord(char)

        # Devanagari
        if 0x0900 <= code <= 0x097F:
            return "hi"

        # Bengali
        if 0x0980 <= code <= 0x09FF:
            return "bn"

        # Gurmukhi
        if 0x0A00 <= code <= 0x0A7F:
            return "pa"

        # Gujarati
        if 0x0A80 <= code <= 0x0AFF:
            return "gu"

        # Tamil
        if 0x0B80 <= code <= 0x0BFF:
            return "ta"

        # Telugu
        if 0x0C00 <= code <= 0x0C7F:
            return "te"

        # Kannada
        if 0x0C80 <= code <= 0x0CFF:
            return "kn"

        # Malayalam
        if 0x0D00 <= code <= 0x0D7F:
            return "ml"

    return None


# ============================================================
# ROMANIZED INDIAN LANGUAGE DETECTION
# ============================================================

def detect_roman_indian_language(text):
    """
    Detect Tanglish / Hinglish only when there is
    reasonably strong evidence.

    This prevents normal English sentences from
    accidentally being classified as Hindi/Tamil.
    """

    text = text.lower().strip()

    words = set(
        re.findall(r"[a-z]+", text)
    )

    # ----------------------------------------
    # Tamil / Tanglish
    # ----------------------------------------

    tamil_words = {
        "panna",
        "mudiyuma",
        "mudiyum",
        "enna",
        "epdi",
        "eppadi",
        "iruka",
        "irukinga",
        "irukiya",
        "panra",
        "panren",
        "pannunga",
        "venum",
        "vendam",
        "illai",
        "illa",
        "aama",
        "sollunga",
        "sollu",
        "inga",
        "anga",
        "romba",
        "nalla",
        "saptiya",
        "saaptiya",
        "sapten",
        "saapten",
        "theriyuma",
        "theriyala",
        "kudunga",
        "kudu",
        "vaanga",
        "pora",
        "poren",
        "poga",
        "vandhu",
        "vantha",
        "vandha",
        "irukku",
        "iruku",
        "yen",
        "yenga",
        "engae",
        "konjam",
        "seekiram",
        "ippo",
        "ippa",
        "naalaikku",
        "innaikku",
        "nethu",
        "enakku",
        "unakku",
        "ungalukku",
        "namma",
        "nanga",
        "naan",
        "nee",
        "neenga",
    }

    # ----------------------------------------
    # Hindi / Hinglish
    # ----------------------------------------

    hindi_words = {
        "kya",
        "kaise",
        "kaisa",
        "kaisi",
        "hain",
        "aap",
        "mujhe",
        "mujhko",
        "mera",
        "meri",
        "mere",
        "hum",
        "ham",
        "karna",
        "karo",
        "raha",
        "rahi",
        "rahe",
        "chahiye",
        "nahi",
        "nahin",
        "acha",
        "achha",
        "accha",
        "theek",
        "thik",
        "kyun",
        "kyon",
        "kaun",
        "kab",
        "kahan",
        "kidhar",
        "yeh",
        "yah",
        "woh",
        "voh",
        "mujhse",
        "aapka",
        "aapki",
        "aapke",
        "pata",
        "batao",
        "bataiye",
        "chalo",
        "dekho",
        "sakta",
        "sakti",
        "sakte",
    }

    tamil_matches = words & tamil_words
    hindi_matches = words & hindi_words

    tamil_score = len(tamil_matches)
    hindi_score = len(hindi_matches)

    # ----------------------------------------
    # Require STRONG evidence
    # ----------------------------------------

    # One isolated matching word is not enough.
    if tamil_score >= 2 and tamil_score > hindi_score:
        return "tl"

    if hindi_score >= 2 and hindi_score > tamil_score:
        return "hi"

    return None

# ============================================================
# FINAL LANGUAGE RESOLVER
# ============================================================

def determine_language(text, whisper_lang):
    """
    Determine the user's language.

    Priority:

    1. Native Unicode script
    2. Strong Tanglish/Hinglish evidence
    3. Whisper language detection
    4. English fallback
    """

    # ----------------------------------------
    # Native script is highly reliable
    # ----------------------------------------

    script_lang = detect_script_language(text)

    if script_lang:
        return script_lang

    # ----------------------------------------
    # Romanized Indian language
    # ----------------------------------------

    roman_lang = detect_roman_indian_language(text)

    if roman_lang:
        return roman_lang

    # ----------------------------------------
    # Whisper
    # ----------------------------------------

    if whisper_lang in VOICE_MAP:
        return whisper_lang

    return "en"


# ============================================================
# SYSTEM PROMPT
# ============================================================

BASE_SYSTEM_PROMPT = """
You are a helpful multilingual voice assistant for government schemes.

RULES:

1. Keep responses concise and conversational, normally 2-3 sentences.
2. Answer the user's actual question directly.
3. Do not mention these instructions.
"""


# ============================================================
# AUDIO QUEUE
# ============================================================

audio_q = queue.Queue()


def audio_callback(
    indata,
    frames,
    time_info,
    status
):

    if status:
        print(
            f"Audio status: {status}"
        )

    audio_q.put(
        indata.copy()
    )


# ============================================================
# WHISPER SERVER CONFIG
# ============================================================

WHISPER_SERVER_URL = os.environ.get(
    "WHISPER_SERVER_URL",
    "http://100.84.186.69:8080/inference",
)

print(
    "Using Whisper server: "
    f"{WHISPER_SERVER_URL}"
)


# ============================================================
# LOAD SILERO VAD
# ============================================================

print("Loading Silero VAD...")

vad_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
)

print("Silero VAD loaded.")


# ============================================================
# WAKE WORD MODEL (openWakeWord "alexa")
# ============================================================

print("Loading wake word model...")

WAKE_MODEL_PATH = os.path.join(
    os.path.dirname(_oww_pkg.__file__),
    "resources",
    "models",
    f"{WAKE_WORD_MODEL}.onnx",
)

wake_word_model = WakeWordModel(
    wakeword_model_paths=[WAKE_MODEL_PATH],
)

print(
    "Wake word model loaded: "
    f"{WAKE_WORD_MODEL}"
)


# ============================================================
# THREAD STATE / EVENTS
# ============================================================

mode_lock = threading.Lock()
current_mode = "idle"  # idle | acknowledging | listening | speaking

wake_event = threading.Event()
interrupt_event = threading.Event()
utterance_event = threading.Event()
utterance_audio = None
stop_event = threading.Event()


# ============================================================
# LLM CLIENT (see llm_service.py)
# ============================================================

# Point to local llama.cpp server (default endpoint)
llm_service.configure_endpoint(
    base_url="http://127.0.0.1:8081/v1",
    model="local",
)

if llm_service.client is None:

    sys.exit(1)


# ============================================================
# CHAT HISTORY
# ============================================================

chat_history = [
    {
        "role": "system",
        "content": BASE_SYSTEM_PROMPT,
    }
]


# ============================================================
# AUDIO CAPTURE WITH VAD + WAKE WORD WORKER
# ============================================================

def play_acceptance_sound():
    """
    Play a soft, barely-audible 'accepted' blip,
    well below normal speech volume.
    """
    sr = 22050

    duration = 0.35

    t = np.linspace(
        0,
        duration,
        int(sr * duration),
        endpoint=False,
    )

    # Single gentle tone with a slow fade in/out
    tone = np.sin(2 * np.pi * 1174 * t)

    envelope = np.minimum(
        t / 0.1,
        (duration - t) / 0.2,
    )
    envelope = np.clip(envelope, 0, 1)

    blip = tone * envelope * 0.4

    # Low volume: a small fraction of normal speech level
    soft_blip = blip * VOLUME_SCALE * 0.5

    sd.play(
        soft_blip.astype("float32"),
        sr,
    )

    sd.wait()


def audio_worker():
    """
    Background thread: owns the single persistent mic stream.

    - idle / speaking  -> runs openWakeWord to detect "alexa"
    - listening        -> runs Silero VAD to capture an utterance
    """

    global current_mode, utterance_audio

    chunk_size = 512

    chunks_per_second = (
        SAMPLE_RATE / chunk_size
    )

    max_silence_chunks = int(
        (SILENCE_PATIENCE_MS / 1000)
        * chunks_per_second
    )

    recorded = []
    triggered = False
    silence_counter = 0

    # openWakeWord expects 16 kHz int16 mono in 1280-sample
    # (80 ms) frames. We accumulate raw float chunks into
    # an int16 buffer and slice frames off the front.
    ww_buffer = np.zeros(0, dtype=np.int16)

    last_wake_time = 0.0

    # When switching to "listening", drain a short period of
    # audio so the echo of the "Yes"/acceptance chime that
    # just played does not get captured as user speech.
    SETTLE_CHUNKS = int(
        0.6 * SAMPLE_RATE / 512
    )
    settle_chunks = 0
    prev_mode = None

    # When leaving "listening", the user's own just-finished
    # speech is still sitting in the audio queue. Drain it and
    # reset the wake model so that tail can't false-trigger
    # "alexa" once we resume wake-word detection.
    WW_SETTLE_CHUNKS = int(
        1.0 * SAMPLE_RATE / 512
    )
    ww_settle_chunks = 0

    try:

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_size,
            callback=audio_callback,
        ):

            while not stop_event.is_set():

                chunk = audio_q.get()

                chunk = chunk.flatten()

                with mode_lock:
                    mode = current_mode

                if mode != prev_mode:

                    # Mode changed.
                    if mode == "listening":

                        recorded = []
                        triggered = False
                        silence_counter = 0
                        settle_chunks = SETTLE_CHUNKS

                    else:

                        # Leaving listening: the captured utterance
                        # (and its trailing audio still in the queue)
                        # must not feed the wake-word detector.
                        ww_buffer = np.zeros(0, dtype=np.int16)
                        ww_settle_chunks = WW_SETTLE_CHUNKS

                        with mode_lock:
                            wake_word_model.reset()

                    prev_mode = mode

                if mode == "listening":

                    if settle_chunks > 0:

                        # Drain echo of the acceptance sound
                        settle_chunks -= 1
                        continue

                    # ------------------------------------
                    # VAD-based utterance capture
                    # ------------------------------------

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

                        recorded.append(
                            chunk
                        )

                        if (
                            silence_counter
                            > max_silence_chunks
                        ):

                            print(
                                "  ...speech finished"
                            )

                            utterance_audio = (
                                np.concatenate(recorded)
                            )

                            recorded = []
                            triggered = False
                            silence_counter = 0

                            with mode_lock:
                                current_mode = "idle"

                            utterance_event.set()

                elif mode in ("idle", "speaking", "processing"):

                    # ------------------------------------
                    # Wake word detection
                    # ------------------------------------

                    if ww_settle_chunks > 0:

                        # Drain the tail of the user's own
                        # just-finished speech.
                        ww_settle_chunks -= 1
                        continue

                    int16 = (
                        np.clip(
                            chunk,
                            -1.0,
                            1.0,
                        )
                        * 32767.0
                    ).astype(np.int16)

                    ww_buffer = (
                        np.concatenate(
                            [ww_buffer, int16]
                        )
                    )

                    while len(ww_buffer) >= 1280:

                        frame = ww_buffer[:1280]

                        ww_buffer = ww_buffer[1280:]

                        predictions = (
                            wake_word_model.predict(
                                frame,
                            )
                        )

                        score = predictions.get(
                            WAKE_WORD_MODEL,
                            0.0,
                        )

                        if score > WAKE_THRESHOLD:

                            now = time.time()

                            if (
                                now - last_wake_time
                                > WAKE_COOLDOWN_S
                            ):

                                last_wake_time = now

                                with mode_lock:
                                    mode_now = current_mode

                                if mode_now == "idle":

                                    print(
                                        "\n🔔 Wake word "
                                        "detected!"
                                    )

                                    wake_event.set()

                                elif (
                                    mode_now
                                    in ("speaking", "processing")
                                ):

                                    print(
                                        "\n⏹️ Interrupt "
                                        "detected!"
                                    )

                                    interrupt_event.set()

                # 'acknowledging' -> just drain audio

    except sd.PortAudioError as e:

        print(
            "⚠️ Microphone error: "
            f"{e}"
        )

    except Exception as e:

        print(
            f"⚠️ Audio worker error: {e}"
        )

    finally:

        print("Audio worker stopped.")


# ============================================================
# TRANSCRIPTION
# ============================================================

def transcribe(audio_np):
    """
    Send recorded audio to the whisper.cpp server
    and return (text, whisper_lang).
    """

    import httpx

    # ----------------------------------------
    # Encode audio as WAV in memory
    # ----------------------------------------

    buf = io.BytesIO()

    sf.write(
        buf,
        audio_np,
        SAMPLE_RATE,
        format="WAV",
        subtype="PCM_16",
    )

    buf.seek(0)

    # ----------------------------------------
    # POST to whisper.cpp server
    # ----------------------------------------

    try:

        with httpx.Client(timeout=300.0) as client:

            response = client.post(
                WHISPER_SERVER_URL,
                files={
                    "file": (
                        "utterance.wav",
                        buf,
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

        print(
            f"⚠️ Whisper server error: {e}"
        )

        return "", "en"

    # ----------------------------------------
    # Parse response
    # ----------------------------------------

    text = result.get(
        "text",
        ""
    ).strip()

    if not text and "segments" in result:
        text = " ".join(
            seg.get("text", "").strip()
            for seg in result.get("segments", [])
            if seg.get("text")
        ).strip()

    if not text:

        return "", "en"

    # verbose_json returns language Probabilities
    whisper_lang = "en"

    raw_lang = str(result.get("language", "")).lower()
    if raw_lang in LANG_NAME_TO_CODE:
        whisper_lang = LANG_NAME_TO_CODE[raw_lang]
    elif raw_lang in VOICE_MAP:
        whisper_lang = raw_lang

    lang_probs = result.get(
        "language_probabilities",
        {}
    )

    if lang_probs and whisper_lang == "en":

        whisper_lang = max(
            lang_probs,
            key=lang_probs.get
        )

    return text, whisper_lang


# ============================================================
# TEXT TO SPEECH
# ============================================================

async def speak(
    text,
    lang_code,
    interruptible=False
):
    """
    Speak `text`. If interruptible, monitor the mic
    wake-word and stop playback early when "alexa"
    is detected. Returns True if interrupted.
    """

    if not text:
        return False

    voice = VOICE_MAP.get(
        lang_code,
        DEFAULT_VOICE
    )

    tmp_path = None

    try:

        communicate = edge_tts.Communicate(
            text,
            voice
        )

        with tempfile.NamedTemporaryFile(
            suffix=".mp3",
            delete=False
        ) as tmp:

            tmp_path = tmp.name

        await communicate.save(
            tmp_path
        )

        data, sr = sf.read(
            tmp_path,
            dtype="float32"
        )

        scaled_data = (
            data * VOLUME_SCALE
        )

        sd.play(
            scaled_data,
            sr
        )

        if not interruptible:

            sd.wait()

            return False

        # ----------------------------------------
        # Interruptible playback: keep draining
        # the output stream while listening for
        # the wake word to stop us.
        # ----------------------------------------

        interrupted = False

        while True:

            stream = sd.get_stream()

            if (
                stream is None
                or not stream.active
            ):
                break

            if interrupt_event.is_set():

                interrupt_event.clear()

                sd.stop()

                interrupted = True

                break

            await asyncio.sleep(0.05)

        return interrupted

    except Exception as e:

        print(
            "⚠️ Couldn't speak reply: "
            f"{e}"
        )

        print(
            f"Reply text: {text}"
        )

        return False

    finally:

        if (
            tmp_path
            and os.path.exists(tmp_path)
        ):

            os.remove(
                tmp_path
            )


# ============================================================
# LLM CALL
# ============================================================

def ask_opencode(
    user_text,
    detected_lang,
    cancel_event=None,
    stream=False,
):
    """
    Send user text to the LLM with language context.

    Each turn, a language prefix is added to the user message
    so the model knows which language to respond in.
    """

    # ----------------------------------------
    # Language name for the prefix
    # ----------------------------------------

    language_name = LANG_NAMES.get(
        detected_lang,
        "English"
    )

    # ----------------------------------------
    # Build the user message with language context
    # ----------------------------------------

    # Add Tanglish-specific guidance if needed
    tanglish_note = ""
    if detected_lang == "tl":
        tanglish_note = (
            " Tanglish means Tamil written in Latin script "
            "mixed with English words (e.g. 'Apply panna mudiyum', "
            "'konjam wait pannungo'). Do NOT use Tamil script."
        )

    user_message = (
        f"[This request is in {language_name}. "
        f"Respond in {language_name}. "
        f"If the user has explicitly asked to speak "
        f"in a different language in this conversation, "
        f"follow that instruction instead.{tanglish_note}]\n\n"
        f"{user_text}"
    )

    # ----------------------------------------
    # RAG retrieval: search for relevant chunks
    # ----------------------------------------

    rag_context = None
    rag_sources = []

    try:
        retriever = get_rag_retriever()
        rag_context, rag_sources = retriever.get_context(
            user_text
        )

        if rag_context:
            print(
                f"📚 RAG: found {len(rag_sources)} "
                f"relevant chunks"
            )

    except Exception as e:
        print(f"⚠️ RAG retrieval failed: {e}")

    # ----------------------------------------
    # Search the web + ask the LLM
    # ----------------------------------------

    return llm_service.generate_response(
        user_text=user_message,
        chat_history=chat_history,
        cancel_event=cancel_event,
        stream=stream,
        rag_context=rag_context,
    )


# ============================================================
# TEXT CLEANING FOR TTS
# ============================================================

def clean_for_tts(text):
    """
    Strip markdown formatting so the TTS engine doesn't
    read symbols like '*' as "asterisk".

    Removes: **bold**, *italic*, `code`, # headers,
    URLs, links, bullet markers and the like.
    """

    if not text:
        return text

    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    cleaned = re.sub(r"\*(.+?)\*", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = cleaned.replace("**", "").replace("__", "")

    return cleaned.strip()


# ============================================================
# JSON RESPONSE
# ============================================================

def format_json_response(
    text
):

    return json.dumps(
        {
            "text": text,
            "audio_script": clean_for_tts(text),
        },
        ensure_ascii=False
    )


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    global current_mode

    print(
        "\n===================================="
    )

    print(
        "Multilingual Voice Assistant"
    )

    print(
        "===================================="
    )

    print(
        "Say 'alexa' to start a conversation. "
        "Say 'alexa' while I speak to interrupt. "
        "Press Ctrl+C to quit."
    )

    # ----------------------------------------
    # Start the persistent mic / wake word
    # background thread.
    # ----------------------------------------

    worker = threading.Thread(
        target=audio_worker,
        daemon=True,
    )

    worker.start()

    try:

        while True:

            # ================================
            # 1. IDLE - wait for wake word
            # ================================

            print(
                "\n🔇 Waiting for wake word... "
                "(say 'alexa')"
            )

            wake_event.clear()

            wake_event.wait()

            wake_event.clear()

            print(
                "\n✅ Wake word detected!"
            )

            # ----------------------------------------
            # Acknowledge: say "yes" + acceptance
            # sound, then listen for the user.
            # ----------------------------------------

            with mode_lock:
                current_mode = "acknowledging"

            asyncio.run(
                speak(
                    "Yes",
                    "en"
                )
            )

            play_acceptance_sound()

            with mode_lock:
                current_mode = "listening"

            print(
                "\n🎤 Listening..."
            )

            # ================================
            # 2. CONVERSATION LOOP
            # ================================

            while True:

                # ----------------------------------------
                # Wait for a captured utterance
                # ----------------------------------------

                utterance_ready = (
                    utterance_event.wait(
                        timeout=WAKE_IDLE_TIMEOUT_S
                    )
                )

                if not utterance_ready:

                    # Nobody spoke in time; back to idle
                    with mode_lock:
                        current_mode = "idle"

                    print(
                        "\n⏳ No speech detected. "
                        "Back to idle."
                    )

                    break

                utterance_event.clear()

                with mode_lock:
                    current_mode = "processing"

                audio_np = utterance_audio

                if (
                    audio_np is None
                    or len(audio_np)
                    < SAMPLE_RATE * 0.3
                ):
                    continue

                # ----------------------------------------
                # Run transcription + LLM search in a
                # background thread so "alexa" can cancel
                # the search while it is still running.
                # ----------------------------------------

                result_box = {}
                cancel_event = threading.Event()
                done_event = threading.Event()

                def _process():

                    try:

                        # WHISPER
                        text, whisper_lang = (
                            transcribe(
                                audio_np
                            )
                        )

                        if not text:

                            result_box["done"] = True
                            return

                        # LANGUAGE RESOLUTION
                        detected_lang = (
                            determine_language(
                                text,
                                whisper_lang
                            )
                        )

                        # HALLUCINATION FILTER
                        clean_text = (
                            text
                            .lower()
                            .strip(".!? ")
                        )

                        if (
                            clean_text
                            in HALLUCINATION_PHRASES
                        ):

                            result_box["ignored"] = True
                            result_box["done"] = True
                            return

                        # OPENCODE ZEN (cancellable, streaming)
                        print("\n🤖 Assistant: ", end="", flush=True)

                        reply_stream = ask_opencode(
                            text,
                            detected_lang,
                            cancel_event=cancel_event,
                            stream=True,
                        )

                        full_reply = ""

                        for chunk in reply_stream:

                            if (
                                cancel_event
                                and cancel_event.is_set()
                            ):
                                break

                            full_reply += chunk
                            sys.stdout.write(chunk)
                            sys.stdout.flush()

                        sys.stdout.write("\n")
                        sys.stdout.flush()

                        result_box["text"] = text
                        result_box["whisper_lang"] = whisper_lang
                        result_box["detected_lang"] = detected_lang
                        result_box["reply"] = full_reply

                    except Exception as e:

                        print(
                            f"⚠️ Processing error: {e}"
                        )

                    finally:

                        result_box["done"] = True
                        done_event.set()

                threading.Thread(
                    target=_process,
                    daemon=True,
                ).start()

                # ----------------------------------------
                # Wait, allowing "alexa" to cancel
                # ----------------------------------------

                cancelled = False

                while not done_event.is_set():

                    if interrupt_event.is_set():

                        interrupt_event.clear()

                        cancel_event.set()

                        cancelled = True

                        with mode_lock:
                            current_mode = "acknowledging"

                        play_acceptance_sound()

                        with mode_lock:
                            current_mode = "listening"

                        print(
                            "\n🎤 Listening..."
                        )

                        break

                    time.sleep(0.05)

                # Cancelled mid-search: re-listen
                if cancelled or not result_box.get("done"):
                    continue

                if result_box.get("ignored"):

                    print(
                        "⚠️ Ignored Whisper "
                        "hallucination."
                    )

                    continue

                if "text" not in result_box:

                    print(
                        "⚠️ No result from processing."
                    )

                    continue

                text = result_box["text"]
                whisper_lang = result_box["whisper_lang"]
                detected_lang = result_box["detected_lang"]
                reply = result_box["reply"]

                # ----------------------------------------
                # DISPLAY
                # ----------------------------------------

                print(
                    f"\n🗣️ You "
                    f"({detected_lang}): {text}"
                )

                if whisper_lang != detected_lang:

                    print(
                        f"   Whisper detected: "
                        f"{whisper_lang}"
                    )

                    print(
                        f"   Corrected to: "
                        f"{detected_lang}"
                    )

                # ----------------------------------------
                # TTS LANGUAGE
                # ----------------------------------------

                reply_lang = detected_lang

                # ----------------------------------------
                # DISPLAY RESPONSE
                # ----------------------------------------

                print(
                    f"🤖 Assistant "
                    f"({reply_lang}): {reply}"
                )

                # ----------------------------------------
                # JSON
                # ----------------------------------------

                sys.stdout.write(
                    format_json_response(
                        reply
                    )
                    + "\n"
                )

                sys.stdout.flush()

                # ----------------------------------------
                # SPEAK (interruptible)
                # ----------------------------------------

                with mode_lock:
                    current_mode = "speaking"

                interrupted = asyncio.run(
                    speak(
                        clean_for_tts(reply),
                        reply_lang,
                        interruptible=True
                    )
                )

                if interrupted:

                    # User said "alexa" while we spoke:
                    # stop, chime, and listen again.
                    with mode_lock:
                        current_mode = "acknowledging"

                    play_acceptance_sound()

                    with mode_lock:
                        current_mode = "listening"

                    print(
                        "\n🎤 Listening..."
                    )

                    continue

                # Reply finished normally; keep the
                # conversation going. The inner loop only
                # returns to idle when nobody speaks for
                # WAKE_IDLE_TIMEOUT_S, so we just continue
                # listening without needing "alexa" again.
                with mode_lock:
                    current_mode = "listening"

                print(
                    "\n🎤 Listening..."
                )

                continue

    except KeyboardInterrupt:

        print(
            "\n\nExiting."
        )

    finally:

        stop_event.set()

        sd.stop()

        worker.join(
            timeout=2.0
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()