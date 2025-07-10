#!/usr/bin/env python3
"""
Orange Pi Alexa Assistant
A custom voice assistant using wake word detection, speech recognition, and ChatGPT
"""

import pyaudio
import wave
import speech_recognition as sr
import requests
import json
import os
import time
import threading
from datetime import datetime

class AlexaAssistant:
    def __init__(self):
        self.is_listening = False
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.record_seconds = 5
        
        # Audio devices (adjust these based on your setup)
        self.input_device_index = None  # Will auto-detect
        self.output_device_index = 3    # Your USB speaker
        
        # API configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # Initialize audio
        self.audio = pyaudio.PyAudio()
        self.recognizer = sr.Recognizer()
        
        print("🤖 Orange Pi Assistant initialized!")
        self.list_audio_devices()
    
    def list_audio_devices(self):
        """List all available audio devices"""
        print("\n📱 Available audio devices:")
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            print(f"  Device {i}: {device_info['name']} - {device_info['maxInputChannels']} in, {device_info['maxOutputChannels']} out")
    
    def detect_wake_word(self, audio_data):
        """Simple wake word detection - replace with better solution later"""
        # For now, we'll use speech recognition to detect "Hey Orange"
        try:
            text = self.recognizer.recognize_google(audio_data).lower()
            wake_words = ["hey orange", "orange pi", "hey assistant"]
            return any(wake_word in text for wake_word in wake_words)
        except:
            return False
    
    def record_audio(self, duration=5):
        """Record audio from microphone"""
        print(f"🎤 Recording for {duration} seconds...")
        
        frames = []
        stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=self.chunk
        )
        
        for i in range(0, int(self.rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        # Save to temporary file
        filename = f"temp_recording_{int(time.time())}.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        return filename
    
    def speech_to_text(self, audio_file):
        """Convert speech to text using Google Speech Recognition"""
        try:
            with sr.AudioFile(audio_file) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                print(f"💬 You said: {text}")
                return text
        except sr.UnknownValueError:
            print("❌ Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"❌ Error with speech recognition: {e}")
            return None
    
    def query_chatgpt(self, text):
        """Send query to ChatGPT API"""
        if not self.openai_api_key:
            return "Please set your OpenAI API key in the environment variables."
        
        try:
            headers = {
                'Authorization': f'Bearer {self.openai_api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'gpt-3.5-turbo',
                'messages': [
                    {'role': 'system', 'content': 'You are a helpful voice assistant. Keep responses concise and conversational.'},
                    {'role': 'user', 'content': text}
                ],
                'max_tokens': 150
            }
            
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                return f"Error: {response.status_code}"
                
        except Exception as e:
            print(f"❌ Error querying ChatGPT: {e}")
            return "Sorry, I couldn't process your request right now."
    
    def text_to_speech(self, text):
        """Convert text to speech and play it"""
        print(f"🔊 Speaking: {text}")
        
        # For now, we'll use espeak (simple TTS)
        # Later we can upgrade to better TTS services
        os.system(f'echo "{text}" | espeak')
    
    def play_audio_file(self, filename):
        """Play audio file through USB speaker"""
        os.system(f'aplay -D plughw:{self.output_device_index},0 {filename}')
    
    def listen_for_wake_word(self):
        """Continuously listen for wake word"""
        print("👂 Listening for wake word...")
        
        while True:
            try:
                # Record short audio snippet
                audio_file = self.record_audio(duration=2)
                
                # Check for wake word
                with sr.AudioFile(audio_file) as source:
                    audio_data = self.recognizer.record(source)
                    
                if self.detect_wake_word(audio_data):
                    print("🎯 Wake word detected!")
                    self.handle_command()
                
                # Clean up temporary file
                os.remove(audio_file)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error in wake word detection: {e}")
                time.sleep(1)
    
    def handle_command(self):
        """Handle voice command after wake word"""
        # Play acknowledgment sound
        print("🎵 Beep! Ready for command...")
        
        # Record user command
        audio_file = self.record_audio(duration=5)
        
        # Convert to text
        text = self.speech_to_text(audio_file)
        
        if text:
            # Get response from ChatGPT
            response = self.query_chatgpt(text)
            
            # Convert response to speech
            self.text_to_speech(response)
        
        # Clean up
        os.remove(audio_file)
    
    def run(self):
        """Main loop"""
        print("🚀 Starting Orange Pi Assistant...")
        print("Say 'Hey Orange' to activate!")
        
        try:
            self.listen_for_wake_word()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
        finally:
            self.audio.terminate()

if __name__ == "__main__":
    assistant = AlexaAssistant()
    assistant.run()