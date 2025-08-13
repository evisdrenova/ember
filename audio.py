import os
import re
import subprocess
import threading
from typing import Iterable, Union
import time
import re

import numpy as np
from kittentts import KittenTTS

_BOUNDARY_RE = re.compile(r'(.+?[\.!?])(\s+|$)')  # greedy minimal up to . ! ?
# Optional mid-phrase soft boundaries to allow earlier speaking if long
_SOFT_BOUNDARY_RE = re.compile(r'(.+?[,;:])(\s+|$)')

# ALSA output (override via env)
OUTPUT_DEVICE = os.getenv("OUTPUT_DEVICE", "plughw:3,0")
START_SENTENCES   = int(os.getenv("TTS_START_SENTENCES", "2"))    # wait for this many sentences
START_MIN_CHARS   = int(os.getenv("TTS_START_MIN_CHARS", "160"))  # or at least this many chars
START_MAX_WAIT    = float(os.getenv("TTS_START_MAX_WAIT", "1.2")) # seconds to wait before starting anyway

# KittenTTS config (override via env)
KITTEN_MODEL = os.getenv("KITTEN_MODEL", "KittenML/kitten-tts-nano-0.1")
KITTEN_VOICE = os.getenv("KITTEN_VOICE", "expr-voice-3-f")
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
    def _pop_complete_sentences(self, buf: str):
        """Return (complete_sentences, remainder). Only cuts at . ! ? boundaries."""
        out = []
        start = 0
        while True:
            m = _BOUNDARY_RE.search(buf, pos=start)
            if not m:
                break
            out.append(m.group(1).strip())
            start = m.end()
        return out, buf[start:]
    
    def _maybe_soft_flush(self, buf: str, emitted_at: float,
                          max_chars: int = 120,
                          max_idle_sec: float = 0.6,
                          min_chars_on_idle: int = 28):
        """
        If we've been idle too long or the buffer is very long,
        allow a 'soft' phrase flush on comma/semicolon/colon,
        or as a last resort flush the whole buffer (to avoid awkward long waits).
        Returns (to_say: list[str], new_buf, new_emitted_at)
        """
        now = time.monotonic()
        to_say = []

        # If buffer is long, try to cut at a soft boundary
        if len(buf) >= max_chars:
            m = _SOFT_BOUNDARY_RE.search(buf[::-1])  # search from end by reversing
            if m:
                cut = len(buf) - m.end(1)
                phrase = buf[:len(buf)-cut].strip()
                if phrase:
                    to_say.append(phrase)
                    return to_say, buf[len(phrase):].lstrip(), now

        # If we've been idle, and buffer has a reasonable size, flush it
        if buf and (now - emitted_at) >= max_idle_sec and len(buf) >= min_chars_on_idle:
            to_say.append(buf.strip())
            return to_say, "", now

        return [], buf, emitted_at

    def stream(self, text_or_iter: Union[_TextLike, _StreamLike]) -> bool:
        with self._stream_lock:
            aplay = None
            try:
                pending = ""
                prebuffer = []       # sentences gathered before starting audio
                started = False
                first_rx = None      # when we first received any text

                def ensure_aplay():
                    nonlocal aplay
                    if aplay is None:
                        aplay = self._open_aplay()

                def speak_sentence(sent: str):
                    audio = self.tts.generate(sent, voice=self.voice)
                    aplay.stdin.write(self._to_int16_pcm(audio))

                for incoming in self._iter_from_maybe_stream(text_or_iter):
                    if first_rx is None:
                        first_rx = time.monotonic()
                    pending += incoming

                    # harvest complete sentences from pending
                    sentences, pending = self._pop_complete_sentences(pending)

                    if not started:
                        # stash sentences until we decide to start
                        prebuffer.extend(sentences)

                        have_enough = (
                            len(prebuffer) >= START_SENTENCES or
                            sum(len(s) for s in prebuffer) >= START_MIN_CHARS or
                            (first_rx is not None and (time.monotonic() - first_rx) >= START_MAX_WAIT and len(prebuffer) > 0)
                        )

                        if have_enough:
                            ensure_aplay()
                            # speak everything we buffered so far
                            for s in prebuffer:
                                speak_sentence(s)
                            prebuffer.clear()
                            started = True
                    else:
                        # already started: speak sentences as they come
                        if sentences:
                            ensure_aplay()
                            for s in sentences:
                                speak_sentence(s)

                # stream ended: flush anything left
                if not started:
                    # never started audio—speak whatever we have (prebuffered sentences + pending)
                    prebuffer.extend(self._pop_complete_sentences(pending)[0])
                    leftover = pending.strip()
                    ensure_aplay()
                    for s in prebuffer:
                        speak_sentence(s)
                    if leftover:
                        speak_sentence(leftover)
                else:
                    # started: speak any final complete sentences and leftover
                    sentences, leftover = self._pop_complete_sentences(pending)
                    ensure_aplay()
                    for s in sentences:
                        speak_sentence(s)
                    if leftover.strip():
                        speak_sentence(leftover.strip())

                if aplay:
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

def tts_say_full(text: str) -> bool:
    """Synthesize the entire text in one go for the smoothest speech."""
    with _STREAMER._stream_lock:
        aplay = None
        try:
            aplay = _STREAMER._open_aplay()
            audio = _STREAMER.tts.generate(text, voice=_STREAMER.voice)  # 24kHz float
            aplay.stdin.write(_STREAMER._to_int16_pcm(audio))
            aplay.stdin.flush()
            aplay.stdin.close()
            rc = aplay.wait()
            if rc != 0:
                print(f"⚠️  aplay exit code: {rc}")
            return True
        except Exception as e:
            print(f"❌ Kitten full TTS failed: {e}")
            try:
                if aplay: aplay.terminate()
            except: pass
            return False
