# Ember

A custom Alexa-like voice assistant running on Orange Pi Zero 3.

## Features

- Wake word detection ("Hey Orange")
- Speech-to-text using Google Speech Recognition
- ChatGPT integration for responses
- Text-to-speech output
- USB microphone and speaker support

## Setup

1. Install dependencies:
   \`\`\`bash
   sudo apt install espeak espeak-data libespeak1 portaudio19-dev python3-dev python3-pip
   pip3 install --user -r requirements.txt
   \`\`\`

2. Set OpenAI API key:
   \`\`\`bash
   export OPENAI_API_KEY="your-api-key-here"
   \`\`\`

3. Run the assistant:
   \`\`\`bash
   python3 main.py
   \`\`\`

## Hardware Requirements

- Orange Pi Zero 3
- USB microphone
- USB speaker
- Internet connection

## Usage

1. Run the script
2. Say "Hey Orange" to activate
3. Speak your command
4. Listen to the response
   EOF
