# audio.py — minimal KittenTTS "say this string" helper

import os
import subprocess
import threading
import numpy as np
from kittentts import KittenTTS

# ---- Config (override via env) ----
OUTPUT_DEVICE = os.getenv("OUTPUT_DEVICE", "plughw:4,0")
KITTEN_MODEL  = os.getenv("KITTEN_MODEL",  "KittenML/kitten-tts-nano-0.1")
KITTEN_VOICE  = os.getenv("KITTEN_VOICE",  "expr-voice-3-f")
SAMPLE_RATE   = 24000  # Kitten outputs 24kHz

# ---- Singleton TTS ----
_TTS = None
_LOCK = threading.Lock()

def preload_tts(model: str | None = None, voice: str | None = None):
    """Load the Kitten model once (optional but recommended at startup)."""
    global _TTS, KITTEN_MODEL, KITTEN_VOICE
    with _LOCK:
        m = model or KITTEN_MODEL
        v = voice or KITTEN_VOICE
        if _TTS is None or m != KITTEN_MODEL or v != KITTEN_VOICE:
            KITTEN_MODEL, KITTEN_VOICE = m, v
            print(f"🐱 Loading KittenTTS: {KITTEN_MODEL} (voice={KITTEN_VOICE})")
            _TTS = KittenTTS(KITTEN_MODEL)
    return _TTS

def _to_int16_pcm(audio) -> bytes:
    arr = np.asarray(audio, dtype=np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16).tobytes()

def tts_say_full(text: str) -> bool:
    """Speak a full string (blocking). Returns True on success."""
    if not text or not text.strip():
        return False

    global _TTS
    if _TTS is None:
        preload_tts()

    p = None
    with _LOCK:  # one synth at a time
        try:
            # Synthesize
            if _TTS is not None:
                audio = _TTS.generate(text.strip(), voice=KITTEN_VOICE)

            # Play via ALSA
            p = subprocess.Popen(
                ["aplay", "-q", "-D", OUTPUT_DEVICE, "-r", str(SAMPLE_RATE), "-f", "S16_LE", "-t", "raw"],
                stdin=subprocess.PIPE
            )
            assert p.stdin is not None
            p.stdin.write(_to_int16_pcm(audio))
            p.stdin.flush()
            p.stdin.close()
            rc = p.wait()
            if rc != 0:
                print(f"⚠️ aplay exit code: {rc}")
            return rc == 0
        except Exception as e:
            print(f"❌ TTS playback failed: {e}")
            try:
                if p: p.terminate()
            except:  # noqa: E722
                pass
            return False
