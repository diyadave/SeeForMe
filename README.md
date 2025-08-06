# Offline Smart Assistant - SeeForMe

## Overview

SeeForMe is a fully offline smart assistant application designed specifically for visually impaired users. The application combines voice interaction, emotion detection, scene analysis, and natural conversation to provide emotional support and environmental awareness. Built with Python Flask and real-time communication via SocketIO.

 💡 The Problem
Most existing solutions for the visually impaired depend on internet-based APIs, making them unreliable in rural or low-connectivity regions. Also, they often lack human-like empathy, fail to provide contextual awareness, and don't support local languages.

🌟 The Vision
To build a fully offline AI companion that not only detects the world around the user but also understands how they feel, talks to them like a friend, and responds appropriately, even when no internet is available.


**Current Status (Aug 1, 2025):**
- ✅ **Working**: Vosk speech recognition, Hybrid AI responses (Gemma3n nano + intelligent patterns), Name extraction, Advanced emotion detection, TTS output, Web interface, **Complete offline memory system with JSON files**, **Face recognition with name learning**, **Automatic camera mapping**, **Emotional continuity across sessions**
- 🔄 **In Progress**: Mobile Kivy wrapper, Real camera integration for mobile deployment
- ⏳ **Planned**: APK/iOS deployment, Enhanced ONNX emotion models

**MAJOR UPDATE**: Switched from PostgreSQL to **fully offline JSON-based memory system** for perfect mobile compatibility and low-end device support.

**Recent Success**: Complete offline memory system implemented! Features emotional continuity ("Yesterday you weren't feeling well..."), face recognition with name learning ("Oh, there is Meera looking at you!"), automatic camera switching, and conversation memory - all using local JSON files. **Ready for local deployment with user's Gemma3n model and DroidCam camera testing.**

**User PC Configuration**: User has Gemma3n:latest (7.5 GB) installed on their PC. When code is downloaded and run locally, it will use their Gemma3n model for responses instead of pattern matching.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture


### Backend Architecture
- **Flask Web Framework**: Serves the web interface and handles HTTP requests
- **SocketIO**: Manages real-time bidirectional communication between frontend and backend
- **Threading-based Processing**: All AI components run in separate threads to prevent blocking
- **Service Layer Pattern**: Modular services for different AI capabilities
- **Offline Memory System**: Local JSON files for complete offline functionality
- **Face Recognition**: OpenCV + face_recognition library for person identification

### Frontend Architecture
- **Web-based Interface**: HTML/CSS/JavaScript with Bootstrap for responsive design
- **Real-time Updates**: JavaScript client communicates with backend via SocketIO
- **Accessibility-first Design**: High contrast, screen reader friendly, keyboard navigation
🧠 Core Features
🧍 Emotion Detection (ONNX + Webcam)
On startup, the assistant scans the user’s face via front camera and detects emotions like happy, sad, neutral, etc.

Empathetic voice responses are generated based on emotion.

Example:
→ “You look upset today. Do you want to talk about it?”

🎙️ Voice Recognition (Offline via Vosk)
Listens for voice commands in English, Hindi, or Gujarati.

Automatically detects spoken language.

Uses Vosk for full offline speech-to-text.

🔊 Text-To-Speech (TTS)
Uses pyttsx3 (offline) for English responses.

For Hindi and Gujarati, gTTS is used only if internet is enabled.

Example:
→ Hindi: "आप थोड़े उदास लग रहे हैं, क्या हुआ?"

💬 Natural Conversation (Gemma 3n via Ollama)
Uses Gemma 3n open-source LLM, running locally via Ollama, for real-time, intelligent replies.

If user talks emotionally or asks questions, Gemma gives comforting and contextual responses.

📸 Camera Switching Logic
If user expresses emotion, the front camera is activated.

If user asks "What’s around me?" or "Someone is coming", the back camera activates.

🔍 Scene, Object & Person Detection (Offline)
Uses:

YOLOv8n for object & person detection

Places365 for scene detection

Example output:
→ “You are in a kitchen. I see a bottle, stove, and a person near the door.”

🧠 Memory System (JSON-based)
Remembers:

User name and face (e.g., Diya, Ramesh)

Past conversations

Last emotion or support given

Memory is stored locally using memory.json.

🔄 Real-time Multithreaded System
All modules run on separate threads for zero lag, smooth interaction, and real-time camera + voice + AI processing.
## Key Components

### 1. Assistant Coordinator (`services/assistant_coordinator.py`)
Central orchestration service that manages all other components and maintains application state.

**Purpose**: Coordinates communication between speech recognition, AI processing, camera management, and TTS
**Key Features**:
- Thread-safe component management
- User context tracking (name, emotion, conversation history)
- Camera mode switching (scene vs emotion detection)
- State management and persistence

### 2. Speech Recognition (`services/speech_handler.py`)
Offline speech recognition using Vosk models for English, Hindi, and Gujarati.

**Purpose**: Convert user speech to text for processing
**Key Features**:
- Multi-language support with auto-detection
- Continuous listening with voice activity detection
- Configurable audio parameters
- Fallback handling for missing dependencies

### 3. Natural Language Processing (`services/gemma_connect.py`)
Integration with Gemma 3n language model via Ollama for conversational AI.

**Purpose**: Generate empathetic, contextual responses to user input
**Key Features**:
- Local Ollama server communication
- Connection pooling and retry logic
- Performance monitoring and caching
- Context-aware prompt engineering for accessibility use cases

### 4. Scene Detection (`services/scene_detector.py`)
Computer vision pipeline combining multiple AI models for environmental understanding.

**Purpose**: Analyze camera feed to describe surroundings and detect people/emotions
**Key Features**:
- YOLOv8n object detection
- Places365 scene classification
- Facial emotion recognition via ONNX models
- Real-time frame processing

### 5. Text-to-Speech (`services/tts_handler.py`)
Multi-language speech synthesis for assistant responses.

**Purpose**: Convert AI-generated text responses to speech
**Key Features**:
- Offline English TTS via pyttsx3
- Online fallback for Hindi/Gujarati via gTTS
- Audio caching and playback management
- Voice customization options

### 6. Name Extraction (`services/name_extractor.py`)
Natural language processing for extracting user names from speech.

**Purpose**: Personalize interactions by learning and using user names
**Key Features**:
- Multi-language name extraction patterns
- Stopword filtering to avoid false positives
- Context-aware name validation

## Data Flow

1. **Audio Input**: User speaks → Vosk converts to text
2. **Intent Analysis**: Text analyzed for emotional context and intent
3. **Camera Switching**: Based on intent, switches between front (emotion) and back (scene) camera
4. **AI Processing**: 
   - Scene: YOLOv8n + Places365 analyze environment
   - Emotion: ONNX model detects user's facial emotion
5. **Context Building**: Results combined with user history and emotional state
6. **Response Generation**: Gemma 3n generates empathetic response using full context
7. **Audio Output**: pyttsx3 converts response to speech
8. **UI Updates**: SocketIO sends real-time status updates to frontend

## External Dependencies

### AI Models (Downloaded locally)
- **Vosk Models**: Speech recognition for multiple languages
- **YOLOv8n**: Object detection model
- **Emotion ONNX Model**: Facial emotion recognition
- **Places365**: Scene classification
- **Gemma 3n**: Language model via Ollama

### Python Libraries
- **Core**: Flask, SocketIO, threading, queue
- **Audio**: pyaudio, pyttsx3, pygame
- **Computer Vision**: OpenCV, ultralytics, torch, onnxruntime
- **ML/AI**: vosk, requests (for Ollama API)
- **Utilities**: pathlib, json, logging

### Hardware Requirements
- **Microphone**: For speech input
- **Camera(s)**: Front camera for emotion detection, back camera for scene analysis
- **Speakers/Headphones**: For TTS output

## Deployment Strategy

### Local Development
- All AI models run locally for complete offline functionality
- Ollama server hosts Gemma 3n model locally
- No internet connection required after initial setup

### Model Setup Process
1. Install Ollama and download Gemma 3n model
2. Download Vosk language models for speech recognition
3. Download YOLOv8n weights
4. Download emotion detection ONNX model
5. Set up Places365 model files

### Performance Considerations
- **Threading**: All AI processing runs in separate threads
- **Memory Management**: Models loaded once and cached
- **Real-time Processing**: Camera feed processed at optimized intervals
- **Fallback Systems**: Graceful degradation when models unavailable

### Accessibility Compliance
- **WCAG 2.1 AA**: High contrast design, keyboard navigation
- **Screen Reader Support**: Semantic HTML, ARIA labels
- **Voice-first Interface**: Primary interaction through speech
- **Multi-language**: Support for English, Hindi, Gujarati

The application is designed to work completely offline once properly configured, ensuring privacy and reliability for users who need consistent access to assistive technology.

SeeForMe/
├── main.py                 # App entry point
├── app/
│   ├── camera_switcher.py  # Front ↔️ Back camera logic
│   ├── emotion_detector.py # ONNX-based emotion detection
│   ├── scene_detector.py   # YOLO + Places365
│   ├── speech_listener.py  # Vosk speech recognition
│   ├── gemma_connect.py    # Gemma 3n via Ollama
│   ├── tts_handler.py      # TTS responses
│   ├── memory_manager.py   # Load/save user memory
│   └── utils.py            # Helper functions
├── assets/                 # Pretrained models
├── memory.json             # User memory file
├── requirements.txt
└── README.md


🙌 Inspiration
This project was inspired by the real challenges faced by the visually impaired — especially in low-resource settings. My goal is to build something that speaks human language, not just machine code — and makes users feel seen, heard, and supported.





## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/diyadave/SeeForMe.git
cd SeeForMe

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
