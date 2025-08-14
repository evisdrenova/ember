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




from __future__ import annotations
import os, sys, struct, json, time, base64, asyncio, subprocess, signal, tempfile
from pathlib import Path
import pvporcupine
import webrtcvad
import websockets
from dotenv import load_dotenv

load_dotenv()

# ---- Your server-side call (kept from your project) ----
from chat import grpc_chat_response  # takes text -> returns string

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ Set OPENAI_API_KEY in env")
    sys.exit(1)

WAKEWORD = "computer"
ALSA_DEVICE = os.getenv("ALSA_DEVICE", "plughw:3,0")
SAMPLE_RATE = 16000   # Porcupine uses 16 kHz; we'll stream 16k mono PCM
FRAME_LEN = 512       # Porcupine frame length (samples)
FRAME_MS = int(1000 * FRAME_LEN / SAMPLE_RATE)  # ~32 ms
VAD_MODE = 2          # 0-3; 3 = most aggressive
END_SIL_MS = 600      # stop after ~600ms of silence
REALTIME_MODEL = "gpt-4o-mini-realtime-preview-2024-12-17"  # fast + realtime
TRANSCRIBE_MODEL = "gpt-4o-transcribe"                      # STT model name

class Ember:
    def __init__(self):
        print("Initializing Ember…")
        try:
            access_key = os.getenv("PICOVOICE_ACCESS_KEY")
            self.porcupine = pvporcupine.create(access_key=access_key, keywords=[WAKEWORD])
            print("✅ Porcupine ready")
        except Exception as e:
            print(f"❌ Porcupine init failed: {e}")
            raise

        self.rate = self.porcupine.sample_rate       # 16000
        self.frame_len = self.porcupine.frame_length # 512
        self.vad = webrtcvad.Vad(VAD_MODE)

        # quick ALSA sanity check
        test_cmd = ["arecord","-D",ALSA_DEVICE,"-f","S16_LE","-r",str(self.rate),"-c","1","-t","wav","-d","1","/tmp/test.wav"]
        try:
            subprocess.run(test_cmd, check=True, capture_output=True)
            Path("/tmp/test.wav").unlink(missing_ok=True)
            print("🎙️ ALSA OK")
        except subprocess.CalledProcessError as e:
            print("❌ ALSA test failed:", e.stderr.decode() if e.stderr else e)
            raise

        print("🎙️  Say 'Computer' to start")

    def _start_arecord_raw(self):
        cmd = ["arecord", "-D", ALSA_DEVICE, "-f", "S16_LE", "-r", str(self.rate), "-c", "1", "-t", "raw"]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _is_speech(self, pcm_bytes: bytes) -> bool:
        # webrtcvad expects 10/20/30ms frames. Our frame is ~32ms; split into two 16ms chunks.
        chunk = 2 * int(0.016 * self.rate)  # samples per 16ms at 16k = 256 samples
        bps = 2  # 16-bit
        ok = False
        for i in range(0, len(pcm_bytes), chunk*bps):
            frame = pcm_bytes[i:i+chunk*bps]
            if len(frame) == chunk*bps and self.vad.is_speech(frame, self.rate):
                ok = True
                break
        return ok

    async def _stt_stream_once(self) -> str:
        """
        Opens a WebSocket session with OpenAI Realtime, streams audio until VAD silence tail,
        and returns the final transcript (string).
        """
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
        uri = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

        async with websockets.connect(uri, additional_headers=headers, ping_interval=20) as ws:
            # 1) Configure session to turn on input_audio transcription
            session_update = {
                "type": "session.update",
                "session": {
                    "input_audio_format": "pcm16",  # we're sending raw 16k Linear PCM
                    "input_audio_transcription": { "model": TRANSCRIBE_MODEL }
                },
                 "turn_detection": {
    "type": "server_vad",
    "threshold": 0.5,
    "prefix_padding_ms": 300,
    "silence_duration_ms": 500,
  }
            }
            await ws.send(json.dumps(session_update))

            # 2) Start the mic -> send audio chunks loop (until VAD says stop)
            arec = self._start_arecord_raw()
            print("🎧 Listening… (streaming to OpenAI)")
            silence_ms = 0
            speaking = False

            try:
                while arec.stdout:
                    buf = arec.stdout.read(self.frame_len * 2)  # 512 samples * 2 bytes
                    if not buf or len(buf) < self.frame_len * 2:
                        break

                    # detect wake word outside (we already did); here we just VAD to stop
                    if self._is_speech(buf):
                        speaking = True
                        silence_ms = 0
                    elif speaking:
                        silence_ms += FRAME_MS
                        if silence_ms >= END_SIL_MS:
                            # stop sending audio and commit
                            break

                    # send audio chunk to input buffer
                    encoded = base64.b64encode(buf).decode("utf-8")
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": encoded
                    }))
            finally:
                # close mic
                try:
                    arec.terminate()
                    arec.wait(timeout=0.5)
                except Exception:
                    pass

            # 3) Commit the audio buffer and request a response (transcript)
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await ws.send(json.dumps({"type": "response.create"}))

            # 4) Listen for events, accumulate transcript text
            final_text = []
            while True:
                msg = await ws.recv()
                evt = json.loads(msg)

                # Common useful events:
                # - response.output_text.delta : text chunk
                # - response.completed         : done
                # - response.error             : failure
                t = evt.get("type")
                if t == "response.output_text.delta":
                    delta = evt.get("delta") or ""
                    final_text.append(delta)
                elif t == "response.completed":
                    break
                elif t == "response.error":
                    raise RuntimeError(f"Realtime error: {evt}")
                # You may also inspect input_audio_transcription.* events if emitted.

            transcript = ("".join(final_text)).strip()
            return transcript

    async def once(self):
        # wake loop: detect "computer" then stream to OpenAI
        # run a lightweight always-on arecord & porcupine
        arec = self._start_arecord_raw()
        print("👂 Wake loop started…")

        try:
            while arec.stdout:
                data = arec.stdout.read(self.frame_len * 2)
                if not data or len(data) < self.frame_len * 2:
                    # restart if arecord hiccups
                    try:
                        arec.terminate(); arec.wait(timeout=0.5)
                    except Exception:
                        pass
                    arec = self._start_arecord_raw()
                    continue

                # Porcupine wake word
                pcm = struct.unpack_from("h" * self.frame_len, data)
                if self.porcupine.process(pcm) >= 0:
                    print("🔵 Wake word detected!")
                    try:
                        # stream STT until silence tail
                        transcript = await self._stt_stream_once()
                    except Exception as e:
                        print(f"❌ STT error: {e}")
                        continue

                    if not transcript:
                        print("😕 No transcript")
                        continue

                    print(f"🗣️ You said: {transcript!r}")

                    # Call your existing gRPC pipeline
                    try:
                        reply = grpc_chat_response(transcript)  # returns plain text
                    except Exception as e:
                        print(f"❌ gRPC error: {e}")
                        continue

                    if reply and reply.strip():
                        print("💬 Assistant:", reply.strip())
                        try:
                            await speak_openai_tts(reply)  # stream TTS to speaker
                        except Exception as e:
                            print(f"❌ TTS error: {e}")
                    else:
                        print("⚠️ Empty reply")

        except KeyboardInterrupt:
            pass
        finally:
            try:
                arec.terminate(); arec.wait(timeout=0.5)
            except Exception:
                pass

# ---- TTS: stream from OpenAI and play on the Pi ----
async def speak_openai_tts(text: str, voice: str = "alloy", audio_format: str = "wav"):
    """
    Streams TTS audio from OpenAI and pipes it to 'aplay' for immediate playback.
    """
    import aiohttp

    url = "https://api.openai.com/v1/audio/speech"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": "gpt-4o-mini-tts",
        "voice": voice,
        "input": text,
        "format": audio_format
    }

    # Start aplay to consume WAV bytes from stdin
    play = subprocess.Popen(["aplay", "-q", "-f", "cd"], stdin=subprocess.PIPE)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                async for chunk in resp.content.iter_chunked(4096):
                    if chunk and play.stdin:
                        play.stdin.write(chunk)
                        play.stdin.flush()
    finally:
        if play.stdin:
            try: play.stdin.close()
            except Exception: pass
        try:
            play.wait(timeout=2)
        except Exception:
            try: play.terminate()
            except Exception: pass

async def main():
    ember = Ember()
    await ember.once()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    asyncio.run(main())

