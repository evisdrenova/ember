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

to connect to the orangepi:

ssh orangepi@192.168.1.17
pw:orangepi

if it doesn't connect the the ip address ahs been changed by the router so plug into the router using an ethernet and then find the new ip

list the networks - `nmcli dev wifi list`
connect to a network - `sudo nmcli dev wifi connect "wifiname" password "your_wifi_password"`

to list audio device - `aplay -l`
look for `UACDemoV10 [UACDemoV1.0], device 0: USB Audio [USB Audio]`
=
