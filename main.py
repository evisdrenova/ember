
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

WAKE_WORD       = "computer"                          # Porcupine built-in keyword
LISTEN_SECONDS  = 4                                # Duration to capture command
INPUT_DEVICE_INDEX: Optional[int] = 3              # Card 3 for input (microphone)
OUTPUT_DEVICE   = "plughw:4,0"                     # Card 4 for output (speaker)
VOSK_MODEL_DIR  = "models/vosk-model-small-en-us-0.15"
PIPER_MODEL     = "models/en_US-amy-low.onnx"
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")      # optional



def play_wave(path: Path):
    """Play WAV file through ALSA using aplay."""
    print(f"🔊 Playing audio file: {path}")
    try:
        subprocess.run(["aplay", "-q", "-D", OUTPUT_DEVICE, str(path)], check=False)
        print("✅ Audio playback completed")
    except Exception as e:
        print(f"❌ Audio playback failed: {e}")

def piper_tts(text: str, wav_out: Path):
    """Generate speech with Piper (try command line first, fallback to espeak)"""
    print(f"🗣️  Generating TTS for: '{text}'")
    print(f"📁 Output file: {wav_out}")
    
    # Try command line piper first
    try:
        proc = subprocess.run(
            ["piper", "--model", PIPER_MODEL, "--output_file", str(wav_out)],
            input=text.encode(),
            check=True,
            capture_output=True
        )
        print("✅ TTS generation completed (Piper)")
        return
    except FileNotFoundError:
        print("⚠️  Piper not found, using espeak-ng...")
    except Exception as e:
        print(f"⚠️  Piper failed: {e}, using espeak-ng...")
    
    # Fallback to espeak-ng - generate WAV file instead of direct audio
    try:
        subprocess.run(
            ["espeak-ng", "-s", "150", "-v", "en", "-w", str(wav_out), text],
            check=True,
            capture_output=True
        )
        print("✅ TTS generation completed (espeak-ng)")
    except Exception as e:
        print(f"❌ espeak-ng failed: {e}")
        # Try festival as last resort
        try:
            # Create a simple text file and use festival
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(text)
                temp_txt = f.name
            
            subprocess.run([
                "text2wave", temp_txt, "-o", str(wav_out)
            ], check=True, capture_output=True)
            os.remove(temp_txt)
            print("✅ TTS generation completed (festival)")
        except Exception as e2:
            print(f"❌ All TTS methods failed: {e2}")
            # Create a beep sound as absolute last resort
            subprocess.run([
                "sox", "-n", str(wav_out), "synth", "0.5", "sine", "800"
            ], check=False, capture_output=True)
            print("⚠️  Created beep sound as fallback")

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
            
            while True:
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