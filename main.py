# #!/usr/bin/env python3
# """
# ========================================================================
#            ORANGE PI – FULLY LOCAL VOICE ASSISTANT (v0.2)
# ========================================================================
# Wake-word  : Picovoice Porcupine  ("jarvis")
# STT        : Vosk small English model
# TTS        : Piper (en_US-amy-low) played via ALSA aplay
# Optional   : ChatGPT completion if OPENAI_API_KEY is exported
# ------------------------------------------------------------------------
# Install deps (Python 3.11/3.12):
#   sudo apt install python3-dev portaudio19-dev libatlas-base-dev \
#                      libsndfile1 espeak-ng alsa-utils
#   python -m venv .venv && source .venv/bin/activate
#   pip install pvporcupine vosk pyaudio piper-tts numpy requests
# Download runtime models once:
#   # Porcupine keyword
#   wget https://github.com/Picovoice/porcupine/raw/master/resources/keyword_files/linux/jarvis_linux.ppn -P models
#   # Vosk STT model (≈ 50 MB)
#   wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
#   unzip vosk-model-small-en-us-0.15.zip -d models
#   # Piper voice (≈ 75 MB)
#   wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en_US/en_US-amy-low.onnx -P models
# ------------------------------------------------------------------------
# """

# from __future__ import annotations
# import os, sys, struct, wave, json, time, tempfile, subprocess, signal
# from pathlib import Path
# from typing import Optional

# import pvporcupine
# import pyaudio
# from vosk import Model, KaldiRecognizer
# import requests

# # ------------------------------------------------------------------ #
# #                CONFIGURATION — tweak to fit your rig               #
# # ------------------------------------------------------------------ #
# WAKE_WORD       = "computer"                          # Porcupine built-in keyword
# LISTEN_SECONDS  = 4                                # Duration to capture command
# INPUT_DEVICE_INDEX: Optional[int] = None           # None = default
# OUTPUT_DEVICE   = "plughw:1,0"                     # ALSA device string for aplay
# VOSK_MODEL_DIR  = "models/vosk-model-small-en-us-0.15"
# PIPER_MODEL     = "models/en_US-amy-low.onnx"
# OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")      # optional


# # ------------------------------------------------------------------ #
# #                              HELPERS                               #
# # ------------------------------------------------------------------ #
# def play_wave(path: Path):
#     """Play WAV file through ALSA using aplay."""
#     subprocess.run(["aplay", "-q", "-D", OUTPUT_DEVICE, str(path)], check=False)

# def piper_tts(text: str, wav_out: Path):
#     """Generate speech with Piper CLI (needs piper in PATH)"""
#     # pipe text via stdin for speed
#     proc = subprocess.run(
#         ["piper", "--model", PIPER_MODEL, "--output_file", str(wav_out)],
#         input=text.encode(),
#         check=True,
#     )

# def chatgpt_response(prompt: str) -> str:
#     if not OPENAI_API_KEY:
#         return f"You said: {prompt}"
#     payload = {
#         "model": "gpt-3.5-turbo",
#         "messages": [
#             {"role": "system", "content": "You are a concise, helpful assistant."},
#             {"role": "user", "content": prompt},
#         ],
#         "max_tokens": 120,
#     }
#     try:
#         r = requests.post(
#             "https://api.openai.com/v1/chat/completions",
#             headers={
#                 "Authorization": f"Bearer {OPENAI_API_KEY}",
#                 "Content-Type": "application/json",
#             },
#             json=payload,
#             timeout=10,
#         )
#         r.raise_for_status()
#         return r.json()["choices"][0]["message"]["content"].strip()
#     except Exception as e:
#         print(f"[ChatGPT error] {e}")
#         return "Sorry, I had trouble thinking just now."

# # ------------------------------------------------------------------ #
# #                           MAIN ASSISTANT                           #
# # ------------------------------------------------------------------ #
# class LocalAssistant:
#     def __init__(self):
#         # Wake-word engine
#         ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
#         if not ACCESS_KEY:
#             raise RuntimeError("Set PICOVOICE_ACCESS_KEY env var first")

#         '''available keywords are = ok google, bumblebee, grasshopper, hey siri, porcupine, terminator, alexa, hey barista, jarvis, computer, americano, pico clock, hey google, grapefruit, blueberry, picovoice'''
#         self.porcupine = pvporcupine.create(
#             access_key=ACCESS_KEY,
#             keywords=["computer"],
#         )
    
#         self.rate      = self.porcupine.sample_rate
#         self.frame_len = self.porcupine.frame_length

#         # STT
#         self.vosk_model = Model(VOSK_MODEL_DIR)
#         self.vosk_rec   = KaldiRecognizer(self.vosk_model, self.rate)

#         # Audio I/O
#         self.pa = pyaudio.PyAudio()
#         self.stream = self.pa.open(
#             rate=self.rate,
#             channels=1,
#             format=pyaudio.paInt16,
#             input=True,
#             frames_per_buffer=self.frame_len,
#             input_device_index=INPUT_DEVICE_INDEX,
#         )
#         print("🎙️  Assistant ready — say 'computer' to start")

#     # -------------------------------------------------------------- #
#     def listen_loop(self):
#         try:
#             while True:
#                 pcm = self.stream.read(self.frame_len, exception_on_overflow=False)
#                 pcm_int16 = struct.unpack_from("h"* self.frame_len, pcm)
#                 if self.porcupine.process(pcm_int16) >= 0:
#                     print("🔵 Wake-word detected!")
#                     self.handle_command()
#         except KeyboardInterrupt:
#             self.cleanup()

#     # -------------------------------------------------------------- #
#     def handle_command(self):
#         # capture spoken command
#         frames: list[bytes] = []
#         chunks = int(self.rate / self.frame_len * LISTEN_SECONDS)
#         print("🎤 Listening for command…")
#         for _ in range(chunks):
#             data = self.stream.read(self.frame_len, exception_on_overflow=False)
#             frames.append(data)
#         # write temp WAV
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
#             wav_path = Path(wf.name)
#             with wave.open(wf, 'wb') as wav_file:
#                 wav_file.setnchannels(1)
#                 wav_file.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16))
#                 wav_file.setframerate(self.rate)
#                 wav_file.writeframes(b"".join(frames))
#         # STT
#         with wave.open(str(wav_path), "rb") as wf:
#             self.vosk_rec.Reset()
#             while True:
#                 buf = wf.readframes(self.frame_len)
#                 if not buf:
#                     break
#                 self.vosk_rec.AcceptWaveform(buf)
#             result = json.loads(self.vosk_rec.FinalResult())
#         os.remove(wav_path)
#         text = result.get("text", "").strip()
#         if not text:
#             print("😕 I didn't catch that.")
#             return
#         print(f"🗣️  You: {text}")

#         # ChatGPT / local reply
#         reply = chatgpt_response(text)
#         print(f"🤖 Assistant: {reply}")

#         # TTS
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
#             tts_path = Path(f.name)
#         piper_tts(reply, tts_path)
#         play_wave(tts_path)
#         os.remove(tts_path)

#     # -------------------------------------------------------------- #
#     def cleanup(self):
#         print("\n🛑 Shutting down…")
#         self.stream.stop_stream()
#         self.stream.close()
#         self.pa.terminate()
#         self.porcupine.delete()
#         sys.exit(0)


# # ------------------------------------------------------------------ #
# if __name__ == "__main__":
#     # Graceful ^C
#     signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
#     assistant = LocalAssistant()
#     assistant.listen_loop()





#!/usr/bin/env python3
"""
========================================================================
           ORANGE PI – FULLY LOCAL VOICE ASSISTANT (v0.2)
========================================================================
Wake-word  : Picovoice Porcupine  ("jarvis")
STT        : Vosk small English model
TTS        : Piper (en_US-amy-low) played via ALSA aplay
Optional   : ChatGPT completion if OPENAI_API_KEY is exported
------------------------------------------------------------------------
Install deps (Python 3.11/3.12):
  sudo apt install python3-dev portaudio19-dev libatlas-base-dev \
                     libsndfile1 espeak-ng alsa-utils
  python -m venv .venv && source .venv/bin/activate
  pip install pvporcupine vosk pyaudio piper-tts numpy requests
Download runtime models once:
  # Porcupine keyword
  wget https://github.com/Picovoice/porcupine/raw/master/resources/keyword_files/linux/jarvis_linux.ppn -P models
  # Vosk STT model (≈ 50 MB)
  wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
  unzip vosk-model-small-en-us-0.15.zip -d models
  # Piper voice (≈ 75 MB)
  wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en_US/en_US-amy-low.onnx -P models
------------------------------------------------------------------------
"""

from __future__ import annotations
import os, sys, struct, wave, json, time, tempfile, subprocess, signal
from pathlib import Path
from typing import Optional

import pvporcupine
import pyaudio
from vosk import Model, KaldiRecognizer
import requests

# ------------------------------------------------------------------ #
#                CONFIGURATION — tweak to fit your rig               #
# ------------------------------------------------------------------ #
WAKE_WORD       = "ember"                          # Porcupine built-in keyword
LISTEN_SECONDS  = 4                                # Duration to capture command
INPUT_DEVICE_INDEX: Optional[int] = None           # None = default
OUTPUT_DEVICE   = "plughw:1,0"                     # ALSA device string for aplay
VOSK_MODEL_DIR  = "models/vosk-model-small-en-us-0.15"
PIPER_MODEL     = "models/en_US-amy-low.onnx"
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")      # optional


# ------------------------------------------------------------------ #
#                              HELPERS                               #
# ------------------------------------------------------------------ #
def play_wave(path: Path):
    """Play WAV file through ALSA using aplay."""
    print(f"🔊 Playing audio file: {path}")
    try:
        subprocess.run(["aplay", "-q", "-D", OUTPUT_DEVICE, str(path)], check=False)
        print("✅ Audio playback completed")
    except Exception as e:
        print(f"❌ Audio playback failed: {e}")

def piper_tts(text: str, wav_out: Path):
    """Generate speech with Piper CLI (needs piper in PATH)"""
    print(f"🗣️  Generating TTS for: '{text}'")
    print(f"📁 Output file: {wav_out}")
    try:
        # pipe text via stdin for speed
        proc = subprocess.run(
            ["piper", "--model", PIPER_MODEL, "--output_file", str(wav_out)],
            input=text.encode(),
            check=True,
        )
        print("✅ TTS generation completed")
    except Exception as e:
        print(f"❌ TTS generation failed: {e}")
        raise

def chatgpt_response(prompt: str) -> str:
    print(f"🤖 Processing prompt: '{prompt}'")
    if not OPENAI_API_KEY:
        print("⚠️  No OpenAI API key found, using echo response")
        return f"You said: {prompt}"
    
    print("📡 Sending request to OpenAI...")
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a concise, helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 120,
    }
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=10,
        )
        r.raise_for_status()
        response = r.json()["choices"][0]["message"]["content"].strip()
        print(f"✅ OpenAI response received: '{response}'")
        return response
    except Exception as e:
        print(f"❌ ChatGPT error: {e}")
        return "Sorry, I had trouble thinking just now."

# ------------------------------------------------------------------ #
#                           MAIN ASSISTANT                           #
# ------------------------------------------------------------------ #
class LocalAssistant:
    def __init__(self):
        print("🚀 Initializing Local Assistant...")
        
        # Wake-word engine
        print("🔑 Checking for Picovoice access key...")
        ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
        if not ACCESS_KEY:
            print("❌ PICOVOICE_ACCESS_KEY environment variable not set!")
            raise RuntimeError("Set PICOVOICE_ACCESS_KEY env var first")
        print("✅ Access key found")

        print("🎯 Initializing Porcupine wake-word detection...")
        '''available keywords are = ok google, bumblebee, grasshopper, hey siri, porcupine, terminator, alexa, hey barista, jarvis, computer, americano, pico clock, hey google, grapefruit, blueberry, picovoice'''
        try:
            self.porcupine = pvporcupine.create(
                access_key=ACCESS_KEY,
                keywords=["computer"],
            )
            print("✅ Porcupine initialized successfully")
        except Exception as e:
            print(f"❌ Porcupine initialization failed: {e}")
            raise
    
        self.rate      = self.porcupine.sample_rate
        self.frame_len = self.porcupine.frame_length
        print(f"📊 Audio settings: {self.rate}Hz, {self.frame_len} frame length")

        # STT
        print(f"🎤 Loading Vosk model from: {VOSK_MODEL_DIR}")
        try:
            self.vosk_model = Model(VOSK_MODEL_DIR)
            self.vosk_rec   = KaldiRecognizer(self.vosk_model, self.rate)
            print("✅ Vosk model loaded successfully")
        except Exception as e:
            print(f"❌ Vosk model loading failed: {e}")
            raise

        # Audio I/O
        print("🎵 Initializing PyAudio...")
        try:
            self.pa = pyaudio.PyAudio()
            print(f"🎙️  Opening audio stream (device: {INPUT_DEVICE_INDEX or 'default'})")
            self.stream = self.pa.open(
                rate=self.rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.frame_len,
                input_device_index=INPUT_DEVICE_INDEX,
            )
            print("✅ Audio stream opened successfully")
        except Exception as e:
            print(f"❌ Audio initialization failed: {e}")
            raise
            
        print("🎙️  Assistant ready — say 'Computer' to start")

    # -------------------------------------------------------------- #
    def listen_loop(self):
        print("👂 Starting listen loop...")
        try:
            while True:
                pcm = self.stream.read(self.frame_len, exception_on_overflow=False)
                pcm_int16 = struct.unpack_from("h"* self.frame_len, pcm)
                keyword_index = self.porcupine.process(pcm_int16)
                if keyword_index >= 0:
                    print(f"🔵 Wake-word detected! (index: {keyword_index})")
                    self.handle_command()
        except KeyboardInterrupt:
            print("\n⚠️  Keyboard interrupt detected")
            self.cleanup()
        except Exception as e:
            print(f"❌ Listen loop error: {e}")
            self.cleanup()

    # -------------------------------------------------------------- #
    def handle_command(self):
        print("🎤 Starting command capture...")
        # capture spoken command
        frames: list[bytes] = []
        chunks = int(self.rate / self.frame_len * LISTEN_SECONDS)
        print(f"🎤 Listening for command for {LISTEN_SECONDS} seconds ({chunks} chunks)...")
        
        for i in range(chunks):
            if i % 10 == 0:  # Progress indicator every 10 chunks
                print(f"📊 Progress: {i}/{chunks} chunks")
            data = self.stream.read(self.frame_len, exception_on_overflow=False)
            frames.append(data)
        
        print("💾 Saving audio to temporary file...")
        # write temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            wav_path = Path(wf.name)
            print(f"📁 Temp file: {wav_path}")
            with wave.open(wf, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16))
                wav_file.setframerate(self.rate)
                wav_file.writeframes(b"".join(frames))
        
        print("🎯 Processing speech with Vosk...")
        # STT
        try:
            with wave.open(str(wav_path), "rb") as wf:
                self.vosk_rec.Reset()
                while True:
                    buf = wf.readframes(self.frame_len)
                    if not buf:
                        break
                    self.vosk_rec.AcceptWaveform(buf)
                result = json.loads(self.vosk_rec.FinalResult())
            print("✅ Speech processing completed")
        except Exception as e:
            print(f"❌ Speech processing failed: {e}")
            os.remove(wav_path)
            return
            
        print(f"🗑️  Cleaning up temp file: {wav_path}")
        os.remove(wav_path)
        
        text = result.get("text", "").strip()
        if not text:
            print("😕 No text detected - I didn't catch that.")
            return
        print(f"🗣️  You said: '{text}'")

        # ChatGPT / local reply
        reply = chatgpt_response(text)
        print(f"🤖 Assistant response: '{reply}'")

        # TTS
        print("🎵 Generating and playing response...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tts_path = Path(f.name)
        
        try:
            piper_tts(reply, tts_path)
            play_wave(tts_path)
        except Exception as e:
            print(f"❌ TTS/Audio error: {e}")
        finally:
            print(f"🗑️  Cleaning up TTS file: {tts_path}")
            if tts_path.exists():
                os.remove(tts_path)
        
        print("✅ Command handling completed\n")

    # -------------------------------------------------------------- #
    def cleanup(self):
        print("\n🛑 Shutting down assistant...")
        try:
            print("🔇 Stopping audio stream...")
            self.stream.stop_stream()
            self.stream.close()
            print("🎵 Terminating PyAudio...")
            self.pa.terminate()
            print("🎯 Cleaning up Porcupine...")
            self.porcupine.delete()
            print("✅ Cleanup completed")
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")
        finally:
            sys.exit(0)


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("=" * 70)
    print("🍊 ORANGE PI LOCAL VOICE ASSISTANT v0.2")
    print("=" * 70)
    
    # Graceful ^C
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    
    try:
        assistant = LocalAssistant()
        assistant.listen_loop()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
