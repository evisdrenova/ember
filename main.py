# from __future__ import annotations
# import os, sys, struct, wave, json, time, tempfile, subprocess, signal
# from pathlib import Path
# from audio import preload_tts,tts_say_full
# from chat import grpc_chat_response
# import pvporcupine
# from vosk import Model, KaldiRecognizer
# from dotenv import load_dotenv
# import whisper


# load_dotenv()

# LISTEN_SECONDS  = 6           
# VOSK_MODEL_DIR  = "models/vosk-model-small-en-us-0.15"

# class Ember:
#     def __init__(self):
#         print("Initializing Ember...")
#         print("-" * 70)
#         try:
#             PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
#             self.porcupine = pvporcupine.create(
#                 access_key=PICOVOICE_ACCESS_KEY,
#                 keywords=["computer"],
#             )
#             print("✅ Porcupine initialized successfully")
#         except Exception as e:
#             print(f"❌ Porcupine initialization failed: {e}")
#             raise

#         self.rate      = self.porcupine.sample_rate
#         self.frame_len = self.porcupine.frame_length
#         print(f"📊 Audio settings: {self.rate}Hz, {self.frame_len} frame length")

#         # STT
#         print(f"🎤 Loading Vosk model from: {VOSK_MODEL_DIR}")
#         try:
#             self.vosk_model = Model(VOSK_MODEL_DIR)
#             self.vosk_rec   = KaldiRecognizer(self.vosk_model, self.rate)
#             print("✅ Vosk model loaded successfully")
#         except Exception as e:
#             print(f"❌ Vosk model loading failed: {e}")
#             raise

#         # Audio I/O - Use ALSA directly since plughw:3,0 works
#         print("🎵 Using direct ALSA recording...")
#         self.alsa_device = "plughw:3,0"

#         # Test ALSA recording works with a proper duration
#         test_cmd = [
#             "arecord", "-D", self.alsa_device, "-f", "S16_LE",
#             "-r", str(self.rate), "-c", "1", "-t", "wav",
#             "-d", "1", "/tmp/test_audio.wav"  # 1 second test
#         ]
        
#         try:
#             print(f"🧪 Testing ALSA recording: {' '.join(test_cmd)}")
#             subprocess.run(test_cmd, check=True, capture_output=True, text=True)
#             print("✅ ALSA recording test successful")
#             if os.path.exists("/tmp/test_audio.wav"):
#                 os.remove("/tmp/test_audio.wav")
#         except subprocess.CalledProcessError as e:
#             print(f"❌ ALSA test failed with exit code {e.returncode}")
#             if e.stderr:
#                 print(f"❌ STDERR: {e.stderr}")
#             if e.stdout:
#                 print(f"❌ STDOUT: {e.stdout}")
#             raise
#         except Exception as e:
#             print(f"❌ ALSA test failed: {e}")
#             raise

#         # Preload Kitten TTS once at startup
#         print("🐱 Preloading KittenTTS...")
#         preload_tts() 
#         print("✅ KittenTTS ready")

#         print("🎙️  Assistant ready — say 'Computer' to start")

#     def listen_loop(self):
#         print("👂 Starting listen loop with ALSA...")

#         # Start continuous recording process
#         arecord_cmd = [
#             "arecord", "-D", self.alsa_device, "-f", "S16_LE",
#             "-r", str(self.rate), "-c", "1", "-t", "raw"
#         ]

#         try:
#             print(f"🎙️  Starting arecord: {' '.join(arecord_cmd)}")
#             self.arecord_process = subprocess.Popen(
#                 arecord_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
#             )

#             while True and self.arecord_process.stdout:
#                 # Read one frame of audio data
#                 audio_data = self.arecord_process.stdout.read(self.frame_len * 2)  # 2 bytes per sample
#                 if len(audio_data) < self.frame_len * 2:
#                     print("⚠️  Audio stream ended (arecord). Restarting capture...")
#                     self.restart_recording()
#                     time.sleep(0.05)
#                     continue
#                 pcm_int16 = struct.unpack_from("h" * self.frame_len, audio_data)
#                 keyword_index = self.porcupine.process(pcm_int16)
#                 if keyword_index >= 0:
#                     print(f"🔵 Wake-word detected! (index: {keyword_index})")
#                     self.handle_command()

#         except KeyboardInterrupt:
#             print("\n⚠️  Keyboard interrupt detected")
#             self.cleanup()
#         except Exception as e:
#             print(f"❌ Listen loop error: {e}")
#             self.cleanup()

#     def handle_command(self):
#         print("🎤 Starting command capture...")

#         # Stop the continuous recording temporarily
#         if hasattr(self, 'arecord_process'):
#             self.arecord_process.terminate()
#             self.arecord_process.wait()

#         # Record command using ALSA directly
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
#             wav_path = Path(wf.name)

#         print(f"🎤 Recording command for {LISTEN_SECONDS} seconds...")
#         record_cmd = [
#             "arecord", "-D", self.alsa_device, "-f", "S16_LE",
#             "-r", str(self.rate), "-c", "1", "-t", "wav",
#             "-d", str(LISTEN_SECONDS), str(wav_path)
#         ]

#         try:
#             subprocess.run(record_cmd, check=True, capture_output=True)
#             print("✅ Command recording completed")
#         except Exception as e:
#             print(f"❌ Command recording failed: {e}")
#             return

#         print("🎯 Processing speech with Vosk...")
#         # STT
#         try:
#             with wave.open(str(wav_path), "rb") as wf:
#                 self.vosk_rec.Reset()
#                 while True:
#                     buf = wf.readframes(self.frame_len)
#                     if not buf:
#                         break
#                     self.vosk_rec.AcceptWaveform(buf)
#                 result = json.loads(self.vosk_rec.FinalResult())
#             print("✅ Speech processing completed")
#         except Exception as e:
#             print(f"❌ Speech processing failed: {e}")
#             os.remove(wav_path)
#             return
        
#         os.remove(wav_path)

#         text = result.get("text", "").strip()
#         if not text:
#             print("😕 No text detected - I didn't catch that.")
#             self.restart_recording()
#             return
#         print(f"🗣️  You said: '{text}'")

#         try:
#             full = grpc_chat_response(text) 
#         except Exception as e:
#             print(f"❌ gRPC error collecting response: {e}")
#             self.restart_recording()
#             return

#         print("the full response", full)
#         if not full or not full.strip():
#             print("⚠️ Empty response from server.")
#             self.restart_recording()
#             return

#         print(f"✅ Complete response received ({len(full)} chars)")
#         print("🗣️ Speaking full response...")
#         try:
#             ok = tts_say_full(full)  
#             if not ok:
#                 print("❌ TTS playback failed")
#         except Exception as e:
#             print(f"❌ TTS/Audio error: {e}")
#         finally:
#             print("✅ Command handling completed\n")
#             # ALWAYS go back to wake-word mode
#             self.restart_recording()

#     def restart_recording(self):
#         """Restart the continuous audio recording process"""
#         # kill old arecord if still around
#         if hasattr(self, 'arecord_process') and self.arecord_process:
#             try:
#                 if self.arecord_process.poll() is None:
#                     self.arecord_process.terminate()
#                     self.arecord_process.wait(timeout=0.5)
#             except Exception:
#                 try: self.arecord_process.kill()
#                 except Exception: pass

#         arecord_cmd = [
#             "arecord", "-D", self.alsa_device, "-f", "S16_LE",
#             "-r", str(self.rate), "-c", "1", "-t", "raw"
#         ]
#         self.arecord_process = subprocess.Popen(
#             arecord_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
#         )

#     def cleanup(self):
#         print("\n🛑 Shutting down assistant...")
#         try:
#             if hasattr(self, 'arecord_process'):
#                 print("🔇 Stopping audio recording process...")
#                 self.arecord_process.terminate()
#                 self.arecord_process.wait()
#             print("🎯 Cleaning up Porcupine...")
#             self.porcupine.delete()
#             print("✅ Cleanup completed")
#         except Exception as e:
#             print(f"⚠️  Cleanup error: {e}")
#         finally:
#             sys.exit(0)

# import whisper

# if __name__ == "__main__":


# # Load the English-only base model
#     m = whisper.load_model("base.en")

#     # Transcribe the file
#     result = m.transcribe("output.wav", language="en", fp16=False)

#     # Print the recognized text
#     print(result["text"])

#     # signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

#     # try:
#     #     # ember = Ember()
#     #     # ember.listen_loop()

#     #     model = whisper.load_model("base")
#     #     result = model.transcribe("output.wav")
#     #     print(result)
#     # except Exception as e:
#     #     print(f"❌ Fatal error: {e}")
#     #     sys.exit(1)

# ember_server_vad.py
from __future__ import annotations
import os, sys, json, struct, signal, asyncio, logging, subprocess, base64
from pathlib import Path
from typing import Optional
import pvporcupine
import websockets
import aiohttp
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from chat import grpc_chat_response  # your unary->server streaming client

# ---------- Config & logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("ember")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    log.error("❌ Set OPENAI_API_KEY in env")
    sys.exit(1)

WAKEWORD     = "computer"
ALSA_DEVICE  = os.getenv("ALSA_DEVICE", "plughw:3,0")
SAMPLE_RATE  = 16000   # Porcupine uses 16k
FRAME_LEN    = 512     # Porcupine frame length (samples)
FRAME_BYTES  = FRAME_LEN * 2  # 16-bit mono
TTS_VOICE    = os.getenv("OPENAI_TTS_VOICE", "alloy")
TTS_FORMAT   = "wav"

# ---------- TTS via OpenAI (streams to aplay) ----------
async def speak_openai_tts(text: str, voice: str = TTS_VOICE, audio_format: str = TTS_FORMAT):
    url = "https://api.openai.com/v1/audio/speech"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {"model": "tts-1", "voice": voice, "input": text, "response_format": audio_format}
    log.info(f"🔊 TTS: {text[:60]}...")

    play = subprocess.Popen(["aplay", "-q", "-f", "cd"], stdin=subprocess.PIPE)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    log.error(f"TTS API error: {resp.status} - {err}")
                    return
                async for chunk in resp.content.iter_chunked(4096):
                    if chunk and play.stdin:
                        play.stdin.write(chunk)
                        play.stdin.flush()
        log.info("✅ TTS playback complete")
    except Exception as e:
        log.error(f"❌ TTS streaming error: {e}", exc_info=True)
    finally:
        try:
            if play.stdin: play.stdin.close()
            play.wait(timeout=2)
        except Exception:
            try: play.terminate()
            except Exception: pass

# ---------- Ember ----------
class Ember:
    def __init__(self):
        log.info("Initializing Ember…")
        # Wake word (Porcupine)
        access_key = os.getenv("PICOVOICE_ACCESS_KEY")
        if not access_key:
            raise RuntimeError("PICOVOICE_ACCESS_KEY not set")
        self.porcupine = pvporcupine.create(access_key=access_key, keywords=[WAKEWORD])
        self.rate      = self.porcupine.sample_rate   # 16000
        self.frame_len = self.porcupine.frame_length  # 512
        log.info("✅ Porcupine ready")

        # ALSA sanity check
        self._test_alsa()
        log.info("🎙️  Say 'Computer' to start")

    def _test_alsa(self):
        cmd = ["arecord", "-D", ALSA_DEVICE, "-f", "S16_LE", "-r", str(self.rate), "-c", "1", "-t", "wav", "-d", "1", "/tmp/test.wav"]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            Path("/tmp/test.wav").unlink(missing_ok=True)
            log.info("🎙️ ALSA OK")
        except subprocess.CalledProcessError as e:
            log.error(f"❌ ALSA test failed: {e.stderr.decode() if e.stderr else e}")
            raise

    def _start_arecord_raw(self):
        cmd = ["arecord", "-D", ALSA_DEVICE, "-f", "S16_LE", "-r", str(self.rate), "-c", "1", "-t", "raw"]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    async def _stt_stream_once(self) -> str:
        """
        Use OpenAI Realtime with SERVER VAD only.
        - Configure correct input format (pcm16, 16kHz)
        - Start a buffer, stream mic frames, stop on server VAD
        - Only commit if we sent enough audio
        """
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }
        uri = "wss://api.openai.com/v1/realtime?intent=transcription"
        MIN_COMMIT_MS = 200  # must be >= 100ms per API error

        async with websockets.connect(uri, additional_headers=headers, ping_interval=20) as ws:
            # 1) Configure the session (use session.update; include sample_rate_hz)
            session_update = {
                "type": "session.update",
                "session": {
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": "gpt-4o-transcribe",
                        "language": "en",
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 600,
                    },
                    "input_audio_noise_reduction": {"type": "near_field"},
                },
            }
            await ws.send(json.dumps(session_update))

            # Wait for ack
            while True:
                evt = json.loads(await ws.recv())
                t = evt.get("type")
                if t in ("session.updated", "transcription_session.updated"):
                    break
                if t == "error":
                    raise RuntimeError(f"Realtime session error: {evt}")

            # 2) Start the audio buffer explicitly
            await ws.send(json.dumps({"type": "input_audio_buffer.start"}))

            # 3) Start mic → append loop (no local VAD)
            arec = self._start_arecord_raw()
            stop_capture = asyncio.Event()
            audio_ms_sent = 0
            final_text = ""

            async def pump_audio():
                loop = asyncio.get_running_loop()
                try:
                    while not stop_capture.is_set():
                        buf = await loop.run_in_executor(None, arec.stdout.read, self.frame_len * 2)  # 512*2=1024 bytes
                        if not buf or len(buf) < self.frame_len * 2:
                            break
                        # append base64 PCM16
                        encoded = base64.b64encode(buf).decode("utf-8")
                        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": encoded}))
                        audio_ms_sent += int(1000 * self.frame_len / self.rate)  # ~32ms per chunk
                finally:
                    with contextlib.suppress(Exception):
                        arec.terminate()
                        arec.wait(timeout=0.5)

            pump_task = asyncio.create_task(pump_audio())

            try:
                # 4) Drive by server VAD & transcription events
                while True:
                    evt = json.loads(await ws.recv())
                    et = evt.get("type")

                    if et == "input_audio_buffer.speech_started":
                        # optional log
                        pass

                    elif et == "input_audio_buffer.speech_stopped":
                        # stop mic and commit if we have enough audio
                        stop_capture.set()
                        await pump_task
                        if audio_ms_sent >= MIN_COMMIT_MS:
                            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                        else:
                            # Not enough audio; end turn without commit
                            return ""

                    elif et in ("input_audio_transcription.partial",):
                        # optional: evt.get("transcript")
                        pass

                    elif et in ("input_audio_transcription.completed", "transcription.final"):
                        final_text = evt.get("transcript") or evt.get("text", "") or ""
                        # ensure mic is stopped, commit if not yet committed and we have audio
                        stop_capture.set()
                        await pump_task
                        if audio_ms_sent >= MIN_COMMIT_MS:
                            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                        break

                    elif et == "error":
                        # If it's the empty-commit error, just return ""
                        err = evt.get("error", {}) or {}
                        if err.get("code") == "input_audio_buffer_commit_empty":
                            return ""
                        raise RuntimeError(f"Realtime error: {evt}")

            finally:
                stop_capture.set()
                with contextlib.suppress(Exception):
                    await pump_task

            return final_text.strip()

    async def once(self):
        """Wake-word loop → server-VAD transcription → gRPC reply → TTS → back to wake loop"""
        arec = self._start_arecord_raw()
        log.info("👂 Wake loop… say 'Computer'")

        try:
            while arec.stdout:
                data = arec.stdout.read(FRAME_BYTES)
                if not data or len(data) < FRAME_BYTES:
                    log.warning("arecord hiccup, restarting…")
                    try:
                        arec.terminate(); arec.wait(timeout=0.5)
                    except Exception:
                        pass
                    arec = self._start_arecord_raw()
                    continue

                pcm = struct.unpack_from("h" * FRAME_LEN, data)
                if self.porcupine.process(pcm) >= 0:
                    log.info("🔵 Wake word detected")
                    try:
                        arec.terminate(); arec.wait(timeout=0.5)
                    except Exception:
                        pass

                    # Transcribe with server VAD
                    try:
                        transcript = await self._stt_stream_once()
                    except Exception as e:
                        log.error(f"❌ STT error: {e}", exc_info=True)
                        arec = self._start_arecord_raw()
                        continue

                    if not transcript:
                        log.info("😕 No transcript")
                        arec = self._start_arecord_raw()
                        continue

                    log.info(f"🗣️ You said: {transcript!r}")

                    # Ask your server (unary→stream, collected on client side)
                    try:
                        reply = grpc_chat_response(transcript)
                    except Exception as e:
                        log.error(f"❌ gRPC error: {e}", exc_info=True)
                        arec = self._start_arecord_raw()
                        continue

                    if reply and reply.strip():
                        log.info(f"💬 Assistant: {reply.strip()}")
                        try:
                            await speak_openai_tts(reply)
                        except Exception as e:
                            log.error(f"❌ TTS error: {e}", exc_info=True)
                    else:
                        log.warning("⚠️ Empty reply from gRPC")

                    # Back to wake loop
                    arec = self._start_arecord_raw()

        except KeyboardInterrupt:
            log.info("Shutting down…")
        finally:
            try:
                arec.terminate(); arec.wait(timeout=0.5)
            except Exception:
                pass

# ---------- main ----------
import contextlib

async def main():
    ember = Ember()
    while True:
        try:
            await ember.once()
        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error(f"Loop error: {e}", exc_info=True)
            await asyncio.sleep(1)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Goodbye!")
        sys.exit(0)
