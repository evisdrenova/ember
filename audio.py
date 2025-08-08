import os
import re
import subprocess
import threading
from typing import Iterable, Union

import numpy as np
from kittentts import KittenTTS

# ALSA output (override via env)
OUTPUT_DEVICE = os.getenv("OUTPUT_DEVICE", "plughw:4,0")

# KittenTTS config (override via env)
KITTEN_MODEL = os.getenv("KITTEN_MODEL", "KittenML/kitten-tts-nano-0.1")
KITTEN_VOICE = os.getenv("KITTEN_VOICE", "expr-voice-2-f")
KITTEN_SR = 24000  # Kitten returns 24kHz audio

_TextLike = Union[str, bytes]
_StreamLike = Iterable[_TextLike]


class _KittenStreamer:
    """Streams text (string or generator) into KittenTTS and plays via ALSA."""

    def __init__(self, model_id: str, voice: str):
        self.model_id = model_id
        self.voice = voice
        print(f"🐱 Loading KittenTTS model: {model_id} (voice={voice})")
        self.tts = KittenTTS(model_id)
        self._stream_lock = threading.Lock()

    @staticmethod
    def _to_int16_pcm(audio) -> bytes:
        """Convert float32 [-1,1] to S16_LE bytes."""
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767.0).astype(np.int16).tobytes()

    @staticmethod
    def _sentence_chunks(text: str):
        """
        Split into sentence-ish chunks to keep latency low.
        Emits on ., !, ?, newline, or when buffer gets long.
        """
        buf = ""
        for piece in str(text).split():
            buf = (buf + " " + piece).strip()
            if re.search(r"[\.!?]\s*$", buf) or len(buf) > 120:
                yield buf
                buf = ""
        if buf.strip():
            yield buf.strip()

    @staticmethod
    def _iter_from_maybe_stream(text_or_iter: Union[_TextLike, _StreamLike]):
        if isinstance(text_or_iter, (str, bytes)):
            yield text_or_iter.decode() if isinstance(text_or_iter, bytes) else text_or_iter
            return
        for chunk in text_or_iter:
            if chunk is None:
                continue
            yield chunk.decode() if isinstance(chunk, bytes) else str(chunk)

    def _open_aplay(self):
        p = subprocess.Popen(
            ["aplay", "-q", "-D", OUTPUT_DEVICE, "-r", str(KITTEN_SR), "-f", "S16_LE", "-t", "raw"],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if not p.stdin:
            raise RuntimeError("aplay stdin not available")
        return p

    def stream(self, text_or_iter: Union[_TextLike, _StreamLike]) -> bool:
        with self._stream_lock:  # one synth at a time on this model
            aplay = None
            try:
                aplay = self._open_aplay()
                pending = ""

                for incoming in self._iter_from_maybe_stream(text_or_iter):
                    pending += incoming
                    for sent in self._sentence_chunks(pending):
                        audio = self.tts.generate(sent, voice=self.voice)  # 24kHz float samples
                        aplay.stdin.write(self._to_int16_pcm(audio))
                        # drop what we just spoke
                        if pending.startswith(sent):
                            pending = pending[len(sent):]
                        else:
                            pending = ""

                if pending.strip():
                    audio = self.tts.generate(pending.strip(), voice=self.voice)
                    aplay.stdin.write(self._to_int16_pcm(audio))

                aplay.stdin.flush()
                aplay.stdin.close()
                rc = aplay.wait()
                if rc != 0:
                    print(f"⚠️  aplay exit code: {rc}")
                return True
            except Exception as e:
                print(f"❌ Kitten streaming failed: {e}")
                try:
                    if aplay:
                        aplay.terminate()
                except:
                    pass
                return False


# ---- Singleton & preload -----------------------------------------------------

_STREAMER = None
_STREAMER_LOCK = threading.Lock()

def preload_tts(model_id: str | None = None, voice: str | None = None):
    """Load the Kitten model once (call at startup)."""
    global _STREAMER
    with _STREAMER_LOCK:
        mid = model_id or KITTEN_MODEL
        v = voice or KITTEN_VOICE
        if _STREAMER is None or _STREAMER.model_id != mid or _STREAMER.voice != v:
            _STREAMER = _KittenStreamer(mid, v)
    return _STREAMER

def tts_stream(text_or_iter: Union[_TextLike, _StreamLike]) -> bool:
    """Stream speech using the preloaded KittenTTS singleton."""
    global _STREAMER
    if _STREAMER is None:
        preload_tts()
    return _STREAMER.stream(text_or_iter)
