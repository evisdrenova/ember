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



import os
import sys
import json
import struct
import signal
import asyncio
import logging
import subprocess
import base64
from pathlib import Path
from typing import Optional

import pvporcupine
import webrtcvad
import websockets
import aiohttp
from dotenv import load_dotenv

load_dotenv()

from chat import grpc_chat_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment checks
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("❌ Set OPENAI_API_KEY in env")
    sys.exit(1)

# Configuration
WAKEWORD = "computer"
ALSA_DEVICE = os.getenv("ALSA_DEVICE", "plughw:3,0")
SAMPLE_RATE = 16000   # Porcupine uses 16 kHz
FRAME_LEN = 512       # Porcupine frame length (samples)
FRAME_MS = int(1000 * FRAME_LEN / SAMPLE_RATE)  # ~32 ms
VAD_MODE = 2          # 0-3; 3 = most aggressive
END_SIL_MS = 600      # stop after ~600ms of silence

class Ember:
    def __init__(self):
        logger.info("Initializing Ember…")
        
        # Initialize Porcupine wake word detection
        try:
            access_key = os.getenv("PICOVOICE_ACCESS_KEY")
            if not access_key:
                raise ValueError("PICOVOICE_ACCESS_KEY not set")
            
            self.porcupine = pvporcupine.create(
                access_key=access_key, 
                keywords=[WAKEWORD]
            )
            logger.info("✅ Porcupine ready")
        except Exception as e:
            logger.error(f"❌ Porcupine init failed: {e}")
            raise

        self.rate = self.porcupine.sample_rate       # 16000
        self.frame_len = self.porcupine.frame_length # 512
        self.vad = webrtcvad.Vad(VAD_MODE)

        # Test ALSA configuration
        self._test_alsa()
        logger.info("🎙️  Say 'Computer' to start")

    def _test_alsa(self):
        """Quick ALSA sanity check"""
        test_cmd = [
            "arecord", "-D", ALSA_DEVICE, "-f", "S16_LE", 
            "-r", str(self.rate), "-c", "1", "-t", "wav", 
            "-d", "1", "/tmp/test.wav"
        ]
        try:
            subprocess.run(test_cmd, check=True, capture_output=True)
            Path("/tmp/test.wav").unlink(missing_ok=True)
            logger.info("🎙️ ALSA OK")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"❌ ALSA test failed: {error_msg}")
            raise

    def _start_arecord_raw(self):
        """Start arecord process for continuous audio capture"""
        cmd = [
            "arecord", "-D", ALSA_DEVICE, "-f", "S16_LE", 
            "-r", str(self.rate), "-c", "1", "-t", "raw"
        ]
        logger.debug(f"Starting arecord: {' '.join(cmd)}")
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _is_speech(self, pcm_bytes: bytes) -> bool:
        """Check if audio contains speech using WebRTC VAD"""
        # WebRTC VAD requires exactly 10, 20, or 30ms frames at 16kHz
        # 30ms at 16kHz = 480 samples = 960 bytes
        frame_duration_ms = 30
        frame_size = int(self.rate * frame_duration_ms / 1000)  # 480 samples
        frame_bytes = frame_size * 2  # 960 bytes (16-bit = 2 bytes per sample)
        
        # Process the audio in 30ms chunks
        speech_detected = False
        for i in range(0, len(pcm_bytes) - frame_bytes + 1, frame_bytes):
            frame = pcm_bytes[i:i + frame_bytes]
            if len(frame) == frame_bytes:
                try:
                    if self.vad.is_speech(frame, self.rate):
                        speech_detected = True
                        break
                except Exception as e:
                    logger.warning(f"VAD error on frame {i//frame_bytes}: {e}")
                    # If VAD fails, try a simpler energy-based detection as fallback
                    # Calculate RMS energy
                    samples = struct.unpack(f"{frame_size}h", frame)
                    energy = sum(s**2 for s in samples) / frame_size
                    if energy > 1000000:  # Threshold for speech energy
                        speech_detected = True
                        break
        
        return speech_detected

    async def _stt_stream_once(self) -> str:
        """
        Opens a WebSocket session with OpenAI Realtime API for transcription,
        streams audio until VAD silence tail, and returns the final transcript.
        """
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
        # Use intent=transcription for transcription-only mode
        uri = "wss://api.openai.com/v1/realtime?intent=transcription"
        
        logger.info("Connecting to OpenAI Realtime API for transcription")
        
        async with websockets.connect(uri, additional_headers=headers, ping_interval=20) as ws:
            # 1) Configure transcription session - wrap config in 'session' parameter
            session_update = {
                "type": "transcription_session.update",
                "session": {
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": "gpt-4o-transcribe",
                        "prompt": "",  # Optional: add context if needed
                        "language": "en"  # Optional: specify language
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                    },
                    "input_audio_noise_reduction": {
                        "type": "near_field"  # Good for close-mic recording
                    }
                }
            }
            
            logger.debug(f"Sending session config: {json.dumps(session_update, indent=2)}")
            await ws.send(json.dumps(session_update))
            
            # Wait for session confirmation
            while True:
                msg = await ws.recv()
                evt = json.loads(msg)
                logger.debug(f"Received event: {evt.get('type')}")
                if evt.get("type") == "transcription_session.updated":
                    logger.info("Session configured successfully")
                    break
                elif evt.get("type") == "session.updated":
                    # Alternative event name
                    logger.info("Session configured successfully")
                    break
                elif evt.get("type") == "error":
                    logger.error(f"Session config error: {evt}")
                    raise RuntimeError(f"Failed to configure session: {evt}")

            # 2) Start streaming audio
            arec = self._start_arecord_raw()
            logger.info("🎧 Listening… (streaming to OpenAI)")
            silence_ms = 0
            speaking = False
            audio_streamed = False

            try:
                while arec.stdout:
                    buf = arec.stdout.read(self.frame_len * 2)  # 512 samples * 2 bytes
                    if not buf or len(buf) < self.frame_len * 2:
                        logger.warning("Incomplete audio frame")
                        break

                    # Local VAD to determine when to stop
                    if self._is_speech(buf):
                        if not speaking:
                            logger.debug("Speech detected locally")
                        speaking = True
                        silence_ms = 0
                    elif speaking:
                        silence_ms += FRAME_MS
                        if silence_ms >= END_SIL_MS:
                            logger.info(f"Local VAD: {silence_ms}ms of silence, stopping")
                            break

                    # Send audio chunk to OpenAI
                    encoded = base64.b64encode(buf).decode("utf-8")
                    audio_msg = {
                        "type": "input_audio_buffer.append",
                        "audio": encoded
                    }
                    await ws.send(json.dumps(audio_msg))
                    audio_streamed = True
                    
            finally:
                # Close mic
                try:
                    arec.terminate()
                    arec.wait(timeout=0.5)
                except Exception as e:
                    logger.warning(f"Error terminating arecord: {e}")

            if not audio_streamed:
                logger.warning("No audio was streamed")
                return ""

            # 3) Commit the audio buffer to trigger final transcription
            logger.debug("Committing audio buffer")
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            # 4) Collect transcription events
            transcript_parts = []
            final_transcript = ""
            
            # Set a timeout for receiving transcription
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    evt = json.loads(msg)
                    event_type = evt.get("type")
                    
                    logger.debug(f"Event: {event_type}")
                    
                    if event_type == "input_audio_transcription.partial":
                        # Partial transcription update
                        partial = evt.get("transcript", "")
                        logger.debug(f"Partial transcript: {partial}")
                        
                    elif event_type == "input_audio_transcription.completed":
                        # Final transcription
                        final_transcript = evt.get("transcript", "")
                        logger.info(f"Final transcript: {final_transcript}")
                        break
                        
                    elif event_type == "transcription.text":
                        # Alternative event name in some versions
                        text = evt.get("text", "")
                        transcript_parts.append(text)
                        
                    elif event_type == "transcription.final":
                        # Final transcription event
                        final_transcript = evt.get("text", "")
                        logger.info(f"Final transcript: {final_transcript}")
                        break
                        
                    elif event_type == "error":
                        logger.error(f"Transcription error: {evt}")
                        raise RuntimeError(f"Transcription error: {evt}")
                        
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for transcription")
                # Use whatever we collected
                if transcript_parts:
                    final_transcript = "".join(transcript_parts)

            return final_transcript.strip()

    async def once(self):
        """Main loop: detect wake word, capture audio, transcribe, and respond"""
        # Start wake word detection loop
        arec = self._start_arecord_raw()
        logger.info("👂 Wake loop started…")
        
        try:
            while arec.stdout:
                data = arec.stdout.read(self.frame_len * 2)
                if not data or len(data) < self.frame_len * 2:
                    logger.warning("Incomplete frame, restarting arecord")
                    # Restart if arecord hiccups
                    try:
                        arec.terminate()
                        arec.wait(timeout=0.5)
                    except Exception:
                        pass
                    arec = self._start_arecord_raw()
                    continue

                # Porcupine wake word detection
                pcm = struct.unpack_from("h" * self.frame_len, data)
                keyword_index = self.porcupine.process(pcm)
                
                if keyword_index >= 0:
                    logger.info("🔵 Wake word detected!")
                    
                    # Stop the wake detection arecord
                    try:
                        arec.terminate()
                        arec.wait(timeout=0.5)
                    except Exception:
                        pass
                    
                    try:
                        # Stream audio to OpenAI for transcription
                        transcript = await self._stt_stream_once()
                    except Exception as e:
                        logger.error(f"❌ STT error: {e}", exc_info=True)
                        # Restart wake detection
                        arec = self._start_arecord_raw()
                        continue

                    if not transcript:
                        logger.info("😕 No transcript received")
                        # Restart wake detection
                        arec = self._start_arecord_raw()
                        continue

                    logger.info(f"🗣️ You said: {transcript!r}")

                    # Call your existing gRPC pipeline
                    try:
                        reply = grpc_chat_response(transcript)  # returns plain text
                    except Exception as e:
                        logger.error(f"❌ gRPC error: {e}", exc_info=True)
                        # Restart wake detection
                        arec = self._start_arecord_raw()
                        continue

                    if reply and reply.strip():
                        logger.info(f"💬 Assistant: {reply.strip()}")
                        try:
                            await speak_openai_tts(reply)  # stream TTS to speaker
                        except Exception as e:
                            logger.error(f"❌ TTS error: {e}", exc_info=True)
                    else:
                        logger.warning("⚠️ Empty reply from gRPC")
                    
                    # Restart wake detection for next interaction
                    arec = self._start_arecord_raw()

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            try:
                arec.terminate()
                arec.wait(timeout=0.5)
            except Exception:
                pass

# ---- TTS: stream from OpenAI and play on the Pi ----
async def speak_openai_tts(text: str, voice: str = "alloy", audio_format: str = "wav"):
    """
    Streams TTS audio from OpenAI and pipes it to 'aplay' for immediate playback.
    """
    url = "https://api.openai.com/v1/audio/speech"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": "tts-1",  # Use correct model name
        "voice": voice,
        "input": text,
        "response_format": audio_format  # Correct parameter name
    }

    logger.info(f"Generating TTS for: {text[:50]}...")
    
    # Start aplay to consume WAV bytes from stdin
    play = subprocess.Popen(["aplay", "-q", "-f", "cd"], stdin=subprocess.PIPE)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"TTS API error: {resp.status} - {error_text}")
                    return
                    
                resp.raise_for_status()
                
                # Stream audio chunks to aplay
                async for chunk in resp.content.iter_chunked(4096):
                    if chunk and play.stdin:
                        play.stdin.write(chunk)
                        play.stdin.flush()
                        
                logger.info("TTS playback complete")
                
    except Exception as e:
        logger.error(f"TTS streaming error: {e}", exc_info=True)
    finally:
        if play.stdin:
            try:
                play.stdin.close()
            except Exception:
                pass
        try:
            play.wait(timeout=2)
        except subprocess.TimeoutExpired:
            logger.warning("aplay timeout, terminating")
            try:
                play.terminate()
            except Exception:
                pass

async def main():
    """Main entry point"""
    ember = Ember()
    
    # Run continuously
    while True:
        try:
            await ember.once()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
            # Continue running despite errors
            await asyncio.sleep(1)

if __name__ == "__main__":
    # Handle SIGINT gracefully
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Goodbye!")
        sys.exit(0)