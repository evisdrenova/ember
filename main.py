
from __future__ import annotations
import os, sys, struct, wave, json, time, tempfile, subprocess, signal
from pathlib import Path
from typing import Optional
from audio import play_wave, piper_tts
from chat import grpc_chat_response
import pvporcupine
from vosk import Model, KaldiRecognizer
from dotenv import load_dotenv

load_dotenv()


LISTEN_SECONDS  = 4                                # Duration to capture command
INPUT_DEVICE_INDEX: Optional[int] = 3              # Card 3 for input (microphone)
VOSK_MODEL_DIR  = "models/vosk-model-small-en-us-0.15"

class Ember:
    def __init__(self):
        print("Initializing Ember...")
        print("-" * 70)
        try:
            PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
            self.porcupine = pvporcupine.create(
                access_key=PICOVOICE_ACCESS_KEY,
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

        # Audio I/O - Use ALSA directly since plughw:3,0 works
        print("🎵 Using direct ALSA recording...")
        self.alsa_device = "plughw:3,0"
        
        # Test ALSA recording works with a proper duration
        test_cmd = [
            "arecord", "-D", self.alsa_device, "-f", "S16_LE", 
            "-r", str(self.rate), "-c", "1", "-t", "wav", 
            "-d", "1", "/tmp/test_audio.wav"  # 1 second test
        ]
        try:
            print(f"🧪 Testing ALSA recording: {' '.join(test_cmd)}")
            result = subprocess.run(test_cmd, check=True, capture_output=True, text=True)
            print("✅ ALSA recording test successful")
            if os.path.exists("/tmp/test_audio.wav"):
                os.remove("/tmp/test_audio.wav")
        except subprocess.CalledProcessError as e:
            print(f"❌ ALSA test failed with exit code {e.returncode}")
            if e.stderr:
                print(f"❌ STDERR: {e.stderr}")
            if e.stdout:
                print(f"❌ STDOUT: {e.stdout}")
            raise
        except Exception as e:
            print(f"❌ ALSA test failed: {e}")
            raise
            
        print("🎙️  Assistant ready — say 'Computer' to start")

    def listen_loop(self):
        print("👂 Starting listen loop with ALSA...")
        
        # Start continuous recording process
        arecord_cmd = [
            "arecord", "-D", self.alsa_device, "-f", "S16_LE",
            "-r", str(self.rate), "-c", "1", "-t", "raw"
        ]
        
        try:
            print(f"🎙️  Starting arecord: {' '.join(arecord_cmd)}")
            self.arecord_process = subprocess.Popen(
                arecord_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            
            while True and self.arecord_process.stdout:
                # Read one frame of audio data
                audio_data = self.arecord_process.stdout.read(self.frame_len * 2)  # 2 bytes per sample
                if len(audio_data) < self.frame_len * 2:
                    print("⚠️  Audio stream ended unexpectedly")
                    break
                
                pcm_int16 = struct.unpack_from("h" * self.frame_len, audio_data)
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

    def handle_command(self):
        print("🎤 Starting command capture...")
        
        # Stop the continuous recording temporarily
        if hasattr(self, 'arecord_process'):
            self.arecord_process.terminate()
            self.arecord_process.wait()
        
        # Record command using ALSA directly
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            wav_path = Path(wf.name)
            
        print(f"🎤 Recording command for {LISTEN_SECONDS} seconds...")
        record_cmd = [
            "arecord", "-D", self.alsa_device, "-f", "S16_LE",
            "-r", str(self.rate), "-c", "1", "-t", "wav",
            "-d", str(LISTEN_SECONDS), str(wav_path)
        ]
        
        try:
            subprocess.run(record_cmd, check=True, capture_output=True)
            print("✅ Command recording completed")
        except Exception as e:
            print(f"❌ Command recording failed: {e}")
            return
        
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
            # Restart continuous recording
            self.restart_recording()
            return
        print(f"🗣️  You said: '{text}'")

        reply = grpc_chat_response(text)
        print(f"🤖 Assistant response: '{reply}'")

        # TTS
        print("🎵 Generating and playing response...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tts_path = Path(f.name)
        
        try:
            piper_tts(reply)
            play_wave(tts_path)
        except Exception as e:
            print(f"❌ TTS/Audio error: {e}")
        finally:
            print(f"🗑️  Cleaning up TTS file: {tts_path}")
            if tts_path.exists():
                os.remove(tts_path)
        
        print("✅ Command handling completed\n")
        
        # Restart continuous recording for wake word detection
        self.restart_recording()
    
    def restart_recording(self):
        """Restart the continuous audio recording process"""
        arecord_cmd = [
            "arecord", "-D", self.alsa_device, "-f", "S16_LE",
            "-r", str(self.rate), "-c", "1", "-t", "raw"
        ]
        self.arecord_process = subprocess.Popen(
            arecord_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def cleanup(self):
        print("\n🛑 Shutting down assistant...")
        try:
            if hasattr(self, 'arecord_process'):
                print("🔇 Stopping audio recording process...")
                self.arecord_process.terminate()
                self.arecord_process.wait()
            print("🎯 Cleaning up Porcupine...")
            self.porcupine.delete()
            print("✅ Cleanup completed")
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")
        finally:
            sys.exit(0)



if __name__ == "__main__":
    # Graceful ^C
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    
    try:
        ember = Ember()
        ember.listen_loop()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)