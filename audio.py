
import os, sys, struct, wave, json, time, tempfile, subprocess, signal
from pathlib import Path


OUTPUT_DEVICE   = "plughw:4,0"                     # Card 4 for output (speaker)
PIPER_MODEL = "models/en_US-amy-medium.onnx"
PIPER_BINARY = "./piper/piper"

def play_wave(path: Path):
    """Play WAV file through ALSA using aplay."""
    print(f"🔊 Playing audio file: {path}")

    try:
        subprocess.run(["aplay", "-q", "-D", OUTPUT_DEVICE, str(path)], check=False)
        print("✅ Audio playback completed")
    except Exception as e:
        print(f"❌ Audio playback failed: {e}")


def piper_tts(text: str):
    """Generate and play speech with Piper streaming (no file creation)"""
    print(f"🗣️  Streaming TTS for: '{text}'")
    
    try:
        # Set up environment with library path
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"./piper:{env.get('LD_LIBRARY_PATH', '')}"
        
        # Stream to aplay directly (no intermediate file)
        piper_process = subprocess.Popen(
            ["./piper/piper", "--model", PIPER_MODEL, "--output-raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        
        # Stream to audio player
        aplay_process = subprocess.Popen(
            ["aplay", "-D", OUTPUT_DEVICE, "-r", "22050", "-f", "S16_LE", "-t", "raw"],
            stdin=piper_process.stdout,
            stderr=subprocess.PIPE
        )

        if  piper_process.stdin and piper_process.stdout:
            # Send text to piper
            piper_process.stdin.write(text.encode())
            piper_process.stdin.close()
            
            # Wait for both processes to complete
            piper_process.stdout.close()
            aplay_return = aplay_process.wait()
            piper_return = piper_process.wait()
        
        if piper_return == 0 and aplay_return == 0:
            print("✅ TTS streaming completed (Piper)")
            return True
        else:
            print(f"⚠️  Streaming failed: piper={piper_return}, aplay={aplay_return}")
            return False
            
    except Exception as e:
        print(f"⚠️  Piper streaming failed: {e}")
        return False